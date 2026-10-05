"""Pinned SSDLite head adaptation and explicit learning from empty images.

Framework objects are injected by the training worker. Importing this module does
not import Torch, construct a detector, load weights, or execute inference.
"""

from copy import deepcopy
from types import MethodType

POLICY = {
    "id": "iris-ssdlite-training-v1",
    "empty_image_hard_negatives": 3,
    "batchnorm": "frozen_running_statistics",
}


def _projections(model, torch):
    """Reject layouts outside the six-scale Torchvision 0.25 SSDLite contract."""
    try:
        classification = model.head.classification_head
        regression = model.head.regression_head
        anchors = model.anchor_generator.num_anchors_per_location()
        if (
            anchors != [6] * 6
            or type(classification.num_columns) is not int
            or classification.num_columns < 2
            or regression.num_columns != 4
            or len(classification.module_list) != 6
            or len(regression.module_list) != 6
        ):
            raise ValueError
        result = []
        for index, (classes, boxes) in enumerate(
            zip(classification.module_list, regression.module_list, strict=True)
        ):
            for block, columns in ((classes, classification.num_columns), (boxes, 4)):
                if not isinstance(block, torch.nn.Sequential) or len(block) != 2:
                    raise ValueError
                projection = block[1]
                if (
                    not isinstance(projection, torch.nn.Conv2d)
                    or projection.kernel_size != (1, 1)
                    or projection.stride != (1, 1)
                    or projection.padding != (0, 0)
                    or projection.dilation != (1, 1)
                    or projection.groups != 1
                    or projection.bias is None
                    or projection.out_channels != anchors[index] * columns
                ):
                    raise ValueError
                if (
                    not isinstance(block[0], torch.nn.Sequential)
                    or not isinstance(block[0][0], torch.nn.Conv2d)
                    or block[0][0].in_channels != projection.in_channels
                    or block[0][0].out_channels != projection.in_channels
                    or block[0][0].groups != projection.in_channels
                ):
                    raise ValueError
            if classes[1].in_channels != boxes[1].in_channels:
                raise ValueError
            result.append((classes, boxes, anchors[index]))
        return classification, result
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Unsupported SSDLite prediction head or anchor layout") from exc


def prepare_ssdlite_head(model, torch, contract, *, trained: bool) -> dict:
    """Retain the pretrained features/box head and map every anchor's class rows."""
    classification, scales = _projections(model, torch)
    classes = len(contract["class_mapping"]) + 1
    if trained:
        if classification.num_columns != classes:
            raise ValueError("Parent SSDLite head does not match its frozen class mapping")
    else:
        if classification.num_columns != 91:
            raise ValueError("The official SSDLite head must contain all 91 COCO slots")
        rows = {
            0: 0,
            **{
                contract["class_mapping"][item["id"]]: item["coco_id"]
                for item in contract["taxonomy"]["classes"]
                if item.get("coco_id") is not None
            },
        }
        for block, _, anchors in scales:
            previous = block[1]
            replacement = torch.nn.Conv2d(
                previous.in_channels,
                anchors * classes,
                1,
                device=previous.weight.device,
                dtype=previous.weight.dtype,
            )
            with torch.no_grad():
                torch.nn.init.normal_(replacement.weight, mean=0.0, std=0.03)
                torch.nn.init.constant_(replacement.bias, 0.0)
                for anchor in range(anchors):
                    for target, source in rows.items():
                        target_row = anchor * classes + target
                        source_row = anchor * 91 + source
                        replacement.weight[target_row].copy_(previous.weight[source_row])
                        replacement.bias[target_row].copy_(previous.bias[source_row])
            block[1] = replacement
        classification.num_columns = classes
    _projections(model, torch)
    return {"head_class_slots": classes}


def configure_ssdlite_training(model, torch) -> dict:
    """Keep native positive loss and mine three background anchors on empty images.

    Torchvision's native SSD mining chooses three negatives per positive anchor.
    With no objects and batch one, that selects none. This explicit training-only
    policy preserves its connected zero box loss and learns from the three hardest
    background classification losses instead. Saved weights retain the native
    inference architecture; there is no inference-time override.
    """
    if getattr(model, "_iris_ssdlite_training_policy", None) is not None:
        raise ValueError("SSDLite training loss has already been configured")
    if model.neg_to_pos_ratio != 3:
        raise ValueError("SSDLite training requires the native three-to-one mining ratio")
    original = model.compute_loss

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        if len(targets) != 1:
            raise ValueError("The frozen SSDLite training policy requires batch size one")
        losses = original(targets, head_outputs, anchors, matched_idxs)
        if targets[0]["boxes"].numel():
            return losses
        if len(matched_idxs) != 1 or torch.any(matched_idxs[0] >= 0).item():
            raise ValueError("An empty SSDLite image unexpectedly matched a positive anchor")
        logits = head_outputs["cls_logits"]
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1:
            raise ValueError("Invalid SSDLite classification logits for an empty image")
        background = torch.zeros(logits.shape[1], dtype=torch.int64, device=logits.device)
        per_anchor = torch.nn.functional.cross_entropy(logits[0], background, reduction="none")
        hard_negatives = min(POLICY["empty_image_hard_negatives"], logits.shape[1])
        return {
            **losses,
            "classification": per_anchor.topk(hard_negatives).values.sum(),
        }

    model.compute_loss = MethodType(compute_loss, model)
    model._iris_ssdlite_training_policy = deepcopy(POLICY)
    return {"training_loss_policy": deepcopy(POLICY)}


def prepare_model(model, contract, *, trained: bool, torch) -> dict:
    """Prepare both adapter surfaces before the caller freezes scopes and BN modes."""
    return {
        **prepare_ssdlite_head(model, torch, contract, trained=trained),
        **configure_ssdlite_training(model, torch),
    }

"""Taxonomy adaptation and loss preparation for the pinned YOLOX-Nano runtime."""

from copy import deepcopy

from iris.yolox_spec import POLICY


def prepare_yolox_head(model, torch, contract, *, trained):
    from iris.yolox_runtime import coco_class_ids

    count = len(contract["class_mapping"])
    head = model.head
    if len(head.cls_preds) != 3 or head.strides != [8, 16, 32]:
        raise ValueError("Unsupported YOLOX prediction layout")
    expected = count if trained else 80
    if head.num_classes != expected or any(
        layer.out_channels != expected for layer in head.cls_preds
    ):
        raise ValueError("YOLOX class head does not match its frozen mapping")
    if not trained:
        coco_rows = {coco_id: index for index, coco_id in enumerate(coco_class_ids())}
        rows = {
            contract["class_mapping"][item["id"]] - 1: coco_rows[item["coco_id"]]
            for item in contract["taxonomy"]["classes"]
            if item.get("coco_id") is not None
        }
        for index, previous in enumerate(head.cls_preds):
            replacement = torch.nn.Conv2d(
                previous.in_channels,
                count,
                1,
                device=previous.weight.device,
                dtype=previous.weight.dtype,
            )
            with torch.no_grad():
                # Match the upstream initial object/class prior for unmapped classes.
                torch.nn.init.constant_(replacement.bias, -4.59511985013459)
                for target, source in rows.items():
                    replacement.weight[target].copy_(previous.weight[source])
                    replacement.bias[target].copy_(previous.bias[source])
            head.cls_preds[index] = replacement
        head.num_classes = count
    return {
        "head_class_slots": count,
        "background_class": False,
        "training_loss_policy": deepcopy(POLICY),
        "head_initialization": (
            "Preserved the trained parent's compatible prediction head"
            if trained
            else "Copied explicitly mapped compact COCO class rows; seeded new rows; "
            "preserved all box and objectness predictions; no background class"
        ),
    }


def training_losses(model, image, boxes, class_mapping, torch, device):
    from iris.yolox_runtime import preprocess

    tensor, ratio = preprocess(image, device)
    # Upstream expects padded [batch,max_objects, class,cx,cy,w,h]. The extra zero
    # row for empty images produces native objectness loss on all background anchors.
    targets = torch.zeros((1, max(1, len(boxes)), 5), dtype=torch.float32, device=device)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box["box"]
        targets[0, index] = torch.tensor(
            [
                class_mapping[box["label"]] - 1,
                (x1 + x2) * ratio / 2,
                (y1 + y2) * ratio / 2,
                (x2 - x1) * ratio,
                (y2 - y1) * ratio,
            ],
            dtype=torch.float32,
            device=device,
        )
    output = model(tensor, targets)
    # total_loss already includes weighted IoU + objectness + class loss. Never sum
    # it a second time alongside component losses or num_fg (a diagnostic ratio).
    return output["total_loss"], {key: output[key] for key in ("iou_loss", "conf_loss", "cls_loss")}


def clip_gradients(parameters, torch):
    """Bound finite batch-one gradients before SGD without concealing invalid values."""
    policy = POLICY["gradient_clipping"]
    norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=policy["max_norm"],
        norm_type=policy["norm_type"],
        error_if_nonfinite=policy["error_if_nonfinite"],
    )
    if not torch.isfinite(norm).item():
        raise ValueError("Nonfinite training gradient norm; no optimizer update was performed")
    return {
        "gradient_norm_before_clip": float(norm.detach()),
        "gradient_clipped": bool(norm > policy["max_norm"]),
        "gradient_max_norm": policy["max_norm"],
    }

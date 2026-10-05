"""Pure, versioned training capabilities; importing this catalog loads no ML runtime."""

from copy import deepcopy

FRCNN = "fasterrcnn_mobilenet_v3_large_320_fpn"
SSDLITE = "ssdlite320_mobilenet_v3_large"
TRAINING_ARCHITECTURES = (FRCNN, SSDLITE)

SCOPE_VERSION = 1
TRAINING_SCOPES = {
    "prediction_head_only": {
        "id": "prediction_head_only",
        "label": "Prediction head only",
        "description": "Adjust the class prediction head while keeping visual features fixed.",
        "trainable_modules": ["roi_heads.box_predictor"],
    },
    "partial_backbone": {
        "id": "partial_backbone",
        "label": "Last backbone stage and detector",
        "description": (
            "Adapt the final visual feature stage, feature pyramid, proposals and ROI heads."
        ),
        "trainable_modules": [
            "backbone.body.13",
            "backbone.body.14",
            "backbone.body.15",
            "backbone.body.16",
            "backbone.fpn",
            "rpn",
            "roi_heads",
        ],
    },
    "full_model": {
        "id": "full_model",
        "label": "All detector layers",
        "description": (
            "Adapt all visual features and detection layers, including early image features."
        ),
        "trainable_modules": ["backbone", "rpn", "roi_heads"],
    },
}


SSDLITE_SCOPES = {
    "prediction_head_only": {
        "id": "prediction_head_only",
        "label": "Detection heads only",
        "description": "Adapt classification and box heads while keeping image features fixed.",
        "trainable_modules": ["head.classification_head", "head.regression_head"],
    },
    "partial_backbone": {
        "id": "partial_backbone",
        "label": "Last feature block and detection heads",
        "description": "Adapt the final MobileNet feature block, extra scales and detection heads.",
        "trainable_modules": ["backbone.features.1", "backbone.extra", "head"],
    },
    "full_model": {
        "id": "full_model",
        "label": "All detector layers",
        "description": "Adapt all visual features and detection heads.",
        "trainable_modules": ["backbone", "head"],
    },
}


def training_scope(scope, architecture=FRCNN):
    if not isinstance(scope, str) or scope not in TRAINING_SCOPES:
        raise ValueError("Choose prediction_head_only, partial_backbone or full_model")
    if architecture not in TRAINING_ARCHITECTURES:
        raise ValueError("Unsupported training architecture")
    return deepcopy((SSDLITE_SCOPES if architecture == SSDLITE else TRAINING_SCOPES)[scope])


def capabilities(architecture):
    if architecture not in TRAINING_ARCHITECTURES:
        return {"training": False, "training_scopes": [], "training_summary": ""}
    return {
        "training": True,
        "training_scopes": [training_scope(scope, architecture) for scope in TRAINING_SCOPES],
        "training_summary": (
            "Lightweight, fixed 320 × 320 detector candidate. Compare quality and target "
            "measurements on your data; smaller weights do not guarantee faster execution."
            if architecture == SSDLITE
            else "Two-stage detector with region proposals. Compare quality and target "
            "measurements on the same held-out images as other candidates."
        ),
    }

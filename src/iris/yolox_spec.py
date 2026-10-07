"""Pure pinned YOLOX contracts; no optional ML runtime is imported here."""

SOURCE_COMMIT = "419778480ab6ec0590e5d3831b3afb3b46ab2aa3"
SOURCE_URL = "https://github.com/Megvii-BaseDetection/YOLOX/tree/" + SOURCE_COMMIT
WEIGHT_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"
)
INPUT_SIZE = 416
INPUT_TRANSFORM = {
    "color": "BGR",
    "tensor_range": [0, 255],
    "exif_transpose": True,
    "fixed_size": [416, 416],
    "dtype": "float32",
    "layout": "NCHW",
    "letterbox": {"alignment": "top_left", "value": 114, "interpolation": "opencv_linear"},
}
NATIVE_FILTERING = {
    "score_threshold": 0.001,
    "nms_iou_threshold": 0.5,
    "max_detections_per_image": 100,
    "score": "objectness multiplied by best class probability",
    "nms": "class_aware",
}
POLICY = {
    "id": "iris-yolox-nano-training-v2",
    "source_commit": SOURCE_COMMIT,
    "input_size": [416, 416],
    "batchnorm": "frozen_running_statistics",
    "loss": "upstream_simota_iou_objectness_classification",
    "augmentation": "none",
    "ema": False,
    "automatic_device_fallback": False,
    "gradient_clipping": {"norm_type": 2.0, "max_norm": 10.0, "error_if_nonfinite": True},
}

"""Class-level contextual relevance priors for an assistive walking scenario.

Values are a manually defined prior R_class in [0, 1] answering: "if this object is
in front of a walking user, how much does it matter?" Obstacles and navigation
landmarks score high; decorative or background objects score low. The prior is
later modulated by the unsupervised scene-context model (see ml/scene_context.py).
"""

RELEVANCE_PRIOR = {
    # navigation hazards / obstacles
    "person": 0.75, "bicycle": 0.85, "car": 0.95, "motorcycle": 0.9, "bus": 0.95,
    "truck": 0.95, "train": 0.9, "dog": 0.7, "cat": 0.4, "horse": 0.7,
    "chair": 0.8, "couch": 0.6, "bench": 0.7, "dining table": 0.7, "bed": 0.5,
    "potted plant": 0.55, "suitcase": 0.7, "backpack": 0.45, "handbag": 0.35,
    "umbrella": 0.4, "skateboard": 0.6, "sports ball": 0.4, "trash can": 0.65,
    "fire hydrant": 0.7, "traffic light": 0.9, "stop sign": 0.95, "parking meter": 0.5,
    "stairs": 0.95, "door": 0.85, "elevator": 0.8, "pole": 0.8, "cone": 0.85,
    "curb": 0.9, "wet floor sign": 0.95, "box": 0.6, "cart": 0.7,
    # background / decorative / low navigation value
    "poster": 0.15, "tv": 0.2, "laptop": 0.2, "mouse": 0.05, "keyboard": 0.05,
    "cell phone": 0.1, "book": 0.1, "clock": 0.15, "vase": 0.2, "bottle": 0.2,
    "cup": 0.2, "remote": 0.05, "scissors": 0.15, "toothbrush": 0.05,
    "refrigerator": 0.35, "microwave": 0.2, "oven": 0.3, "sink": 0.35, "toilet": 0.5,
    "painting": 0.1, "window": 0.2, "plant": 0.3, "lamp": 0.25,
}

DEFAULT_RELEVANCE = 0.3

# rough physical height (m) used to convert bbox height into a distance-like signal
TYPICAL_HEIGHT_M = {
    "person": 1.7, "chair": 0.9, "door": 2.1, "car": 1.5, "bicycle": 1.0, "dog": 0.5,
    "trash can": 0.9, "bench": 0.9, "table": 0.75, "dining table": 0.75, "couch": 0.9,
    "backpack": 0.45, "suitcase": 0.7, "potted plant": 0.6, "stairs": 1.5,
    "fire hydrant": 0.8, "stop sign": 0.75, "traffic light": 1.0, "cone": 0.7, "pole": 2.5,
}


def relevance_prior(label: str) -> float:
    return RELEVANCE_PRIOR.get(label, DEFAULT_RELEVANCE)

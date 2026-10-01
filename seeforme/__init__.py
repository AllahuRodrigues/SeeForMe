"""SeeForMe: context-aware visual assistance for blind and low-vision users.

The package is organised around the pipeline described in the CPSC 4900 proposal:

    camera -> preprocessing -> detection + OCR -> tracking / temporal memory
           -> candidate observations -> cue selection (baselines or CACS)
           -> speech / debug overlay

Sub-packages
------------
perception   YOLO object detection, EasyOCR text reading, frame preprocessing
tracking     lightweight IoU multi-object tracker with per-track announcement memory
cues         feature extraction (C, R, N, P, T, D), CACS, and baseline policies
ml           supervised text-importance classifier, supervised cue-priority model,
             unsupervised scene-context clustering
sim          synthetic scenario generator with ground-truth important events
eval         cue-selection metrics, experiment runner, ablations
feedback     text-to-speech output
ui           OpenCV debugging overlay
"""

__version__ = "0.1.0"

# YOLO-World prompts on 59 frames from 30 EgoBlind videos that mention a door or stairs

Detector `yolov8s-worldv2.pt`, imgsz 640, confidence floor 0.05; frames at t_q - 3 s and t_q (`scripts/probe_openvocab_prompts.py`, 2026-10-09).

| prompt | detections at any confidence | at the caching threshold (0.25) |
|---|---|---|
| door | 3 | 0 |
| doorway | 14 | 1 |
| open door | 9 | 0 |
| entrance | 1 | 0 |
| exit | 0 | 0 |
| glass door | 8 | 0 |
| sliding door | 52 | 4 |
| gate | 1 | 0 |
| elevator door | 42 | 9 |
| stairs | 0 | 0 |
| staircase | 2 | 0 |
| steps | 3 | 0 |
| stairway | 2 | 1 |
| escalator | 18 | 3 |
| handrail | 5 | 0 |
| wall | 53 | 3 |
| person | 169 | 60 |

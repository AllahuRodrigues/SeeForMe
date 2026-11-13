Detector comparison on review frames inside train/finetune event windows (confidence >= 0.35); device mps for HF models

| detector | licence | events scored | seen | event recall | detections / frame | ms / frame |
|---|---|---|---|---|---|---|
| yolo11 | AGPL-3.0 (Ultralytics) | 195 | 163 | 0.836 | 3.9 | 27 |
| rtdetr_r18 | Apache-2.0 | 195 | 176 | 0.903 | 7.7 | 270 |
| dfine_s | Apache-2.0 | 195 | 181 | 0.928 | 14.6 | 285 |

Per label (seen / events): yolo11: person 84/84, car 52/59, motorcycle 34/34, bicycle 19/20, truck 16/19, dining table 3/16, chair 5/8, knife 5/6, scissors 1/6, sink 4/5; rtdetr_r18: person 84/84, car 56/59, motorcycle 34/34, bicycle 20/20, truck 19/19, dining table 10/16, chair 6/8, knife 6/6, scissors 6/6, sink 5/5; dfine_s: person 84/84, car 57/59, motorcycle 34/34, bicycle 20/20, truck 19/19, dining table 13/16, chair 6/8, knife 6/6, scissors 6/6, sink 5/5

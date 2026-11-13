Segmentation probe (nvidia/segformer-b2-finetuned-ade-512-512); concept seen if its classes cover >= 2% of the path region in a window frame

| concept | seen on its not-detectable events | base rate on events not about it |
|---|---|---|
| door | 7/24 (29%) | 9% (n=259) |
| stairs | 5/9 (56%) | 10% (n=276) |
| pole | 1/9 (11%) | 14% (n=280) |
| fence | 5/6 (83%) | 22% (n=281) |
| wall | 4/4 (100%) | 72% (n=285) |
| sign | 2/3 (67%) | 21% (n=290) |
| shelf | 0/3 (0%) | 20% (n=289) |
| animal | 0/3 (0%) | 7% (n=290) |
| tree | 1/2 (50%) | 17% (n=287) |

Walkable share of the path region (floor, sidewalk, road, path, ...):

| group | n | mean | median |
|---|---|---|---|
| other_speak_events | 259 | 0.42 | 0.46 |
| no_object_named | 34 | 0.35 | 0.26 |
| user_confirmed_quiet | 20 | 0.28 | 0.22 |

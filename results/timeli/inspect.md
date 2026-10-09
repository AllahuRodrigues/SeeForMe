# TIMELI annotation files: format and counts

One record per frame at 1 FPS; `id` = seconds since the start of the session video; the instruction for that second is `QA[0].a` (JSON `{reason, instruction}` for CARLA, plain text for the real set); silence = instruction `None` with reason `remain_silent` / `remain_silent_in_junction`; `history` = the 15 previous seconds, oldest first. See the module docstring for the generator rules.

| split | file | records (frames) | videos | frames / video | spoken share | instr. / min | words / instr. | silence runs (mean s, longest) | gap between instr. (median s) | junction share |
|---|---|---|---|---|---|---|---|---|---|---|
| train | train_labels.json (360 MB) | 56281 | 269 | 51-705 (mean 209.2) | 0.207 | 12.4 | 11.63 +/- 4.66 (max 32) | 11776 (3.79, 24) | 4.0 | 0.233 |
| test | test_labels.json (70 MB) | 10820 | 20 | 242-705 (mean 541) | 0.22 | 13.23 | 11.19 +/- 4.21 (max 28) | 2397 (3.52, 28) | 4 | 0.128 |
| real | real_test.json (5 MB) | 1059 | 11 | 47-165 (mean 96.3) | 0.291 | 17.45 | 9.46 +/- 4.27 (max 20) | 264 (2.84, 8) | 3 | 0.0 |

## Reasons (why the generator spoke or stayed silent)

| split | None | constant_instruction | constantly_instruct | direction_change | enter_junction | exit_junction | obstacle_in_front | remain_silent | remain_silent_in_junction | stop |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 0 | 2446 | 0 | 3814 | 890 | 884 | 3588 | 37536 | 7114 | 9 |
| test | 0 | 419 | 0 | 811 | 129 | 128 | 898 | 7699 | 736 | 0 |
| real | 1 | 83 | 1 | 0 | 6 | 6 | 153 | 784 | 25 | 0 |

## Fields (first record)

### train

| field | type | example |
|---|---|---|
| `id` | int | `16` |
| `video_id` | str | `"data_2024_10_25_03_04_21_session_2_Town05_wk1"` |
| `video` | str | `"videos/videos_data_2024_10_25_03_04_21_session_2_Town05_wk1_0_16.mp4"` |
| `conversations` | list | `[{"from": "human", "value": "<image>\nYou are guiding a blind person. The blind person needs to approach the g...` |
| `QA` | list | `[{"q": "<image>\nYou are guiding a blind person. The blind person needs to approach the goal: [x,y]=[-0.0, 1.7...` |
| `history` | list | `["Okay, go straight. There's a fence and traffic sign on your left.", "None", "None", "None", "None", "None", ...` |
| `goal_coord` | list | `[-0.0, 1.71]` |
| `is_junction` | bool | `false` |
| `reformatted_past_instruction` | str | `"\ninstruction 15 seconds ago was: Okay, go straight. There's a fence and traffic sign on your left.\ninstruct...` |
| `description_cluster` | dict | `{"FENCE": ["front"]}` |
| `action` | str | `"walk straight"` |
| `image_name` | str | `"image_2024_10_25_03_04_21_session_2_Town05_wk1_0_16.jpg"` |
| `image` | str | `"images/image_2024_10_25_03_04_21_session_2_Town05_wk1_0_16.jpg"` |
| `meta` | dict | `{"id": "000000000160", "height": 1080, "width": 1920, "date_captured": "Fri Oct 25 03:50:43 2024", "town": "To...` |
| `object_detection` | list | `[]` |
| `speaking_reason` | str | `"constant_instruction"` |
| `conveyed_obstacles` | dict | `{}` |

Actions: {'walk straight': 45845, 'turn slightly right': 4613, 'turn slightly left': 4586, 'stop': 571, 'turn around': 295, 'turn right': 187, 'turn left': 184}. Conveyed obstacles: {'pole': 5274, 'pedestrian': 4986, 'building': 4830, 'vegetation': 3487, 'object': 2622, 'wall': 2363, 'fence': 2308, 'vehicle': 1082, 'traffic sign': 469} (directions {'right': 11661, 'front': 4686, 'left': 11523}). Scene objects (`description_cluster`): {'BUILDING': 11291, 'FENCE': 9574, 'POLE': 9430, 'PEDESTRIAN': 7965, 'VEGETATION': 7628, 'OBJECT': 5610, 'WALL': 2994, 'VEHICLE': 1833, 'TRAFFIC_SIGN': 891}. Detector labels (`object_detection`): {'person': 11215, 'banana': 6334, 'car': 3278, 'bench': 2574, 'traffic light': 2477, 'potted plant': 1476, 'motorcycle': 896, 'bicycle': 568, 'truck': 567, 'handbag': 414, 'stop sign': 263, 'chair': 256, 'backpack': 204, 'skateboard': 173, 'book': 89}. Weather: {'WetCloudyNoon': 11622, 'WetNoon': 11589, 'ClearNoon': 10935, 'HardRainNoon': 9519, 'ClearSunset': 7063, 'SoftRainSunset': 5553}; town: {'Town10HD': 28380, 'Town05': 27901}. Silence runs by length (s): {'1': 3125, '10+': 163, '2-3': 3380, '4-6': 2888, '7-9': 2220}. First-frame reason: {'constant_instruction': 12, 'remain_silent': 173, 'remain_silent_in_junction': 35, 'obstacle_in_front': 21, 'direction_change': 20, 'exit_junction': 3, 'enter_junction': 5}. Reason / instruction inconsistencies: none. `history[-1]` equals the previous frame's instruction in 1.0 of frames.

### test

| field | type | example |
|---|---|---|
| `id` | int | `16` |
| `video_id` | str | `"data_2024_11_01_11_56_10_fps_10_sample_10_test_session_0_Town02_wk1"` |
| `video` | str | `"videos/videos_data_2024_11_01_11_56_10_fps_10_sample_10_test_session_0_Town02_wk1_0_16.mp4"` |
| `conversations` | list | `[{"from": "human", "value": "<image>\nYou are guiding a blind person. The blind person needs to approach the g...` |
| `QA` | list | `[{"q": "<image>\nYou are guiding a blind person. The blind person needs to approach the goal: [x,y]=[0.0, 1.08...` |
| `history` | list | `["Okay, go straight.", "None", "None", "None", "None", "None", "None", "Go to your two and keep walking.", "No...` |
| `goal_coord` | list | `[0.0, 1.08]` |
| `is_junction` | bool | `false` |
| `reformatted_past_instruction` | str | `"\ninstruction 15 seconds ago was: Okay, go straight.\ninstruction 14 seconds ago was: None\ninstruction 13 se...` |
| `description_cluster` | dict | `{"BUILDING": ["front"]}` |
| `action` | str | `"walk straight"` |
| `image_name` | str | `"image_2024_11_01_11_56_10_fps_10_sample_10_test_session_0_Town02_wk1_0_16.jpg"` |
| `image` | str | `"images/image_2024_11_01_11_56_10_fps_10_sample_10_test_session_0_Town02_wk1_0_16.jpg"` |
| `meta` | dict | `{"id": "000000000160", "height": 1080, "width": 1920, "date_captured": "Sat Nov  2 00:13:39 2024", "town": "To...` |
| `object_detection` | list | `[]` |
| `speaking_reason` | str | `"remain_silent"` |
| `conveyed_obstacles` | dict | `{}` |

Actions: {'walk straight': 8475, 'turn slightly left': 1095, 'turn slightly right': 1091, 'turn around': 49, 'turn right': 47, 'turn left': 38, 'stop': 25}. Conveyed obstacles: {'pedestrian': 1119, 'pole': 1095, 'building': 966, 'object': 643, 'vegetation': 447, 'fence': 376, 'wall': 283, 'vehicle': 135, 'traffic sign': 42, 'guard rail': 3} (directions {'front': 1134, 'right': 2121, 'left': 1925}). Scene objects (`description_cluster`): {'POLE': 2393, 'BUILDING': 2271, 'PEDESTRIAN': 1966, 'OBJECT': 1455, 'FENCE': 1046, 'VEGETATION': 1006, 'WALL': 923, 'VEHICLE': 244, 'TRAFFIC_SIGN': 100, 'GUARD_RAIL': 19}. Detector labels (`object_detection`): {'person': 2234, 'banana': 1535, 'car': 549, 'bench': 483, 'potted plant': 202, 'traffic light': 151, 'fire hydrant': 111, 'truck': 105, 'motorcycle': 102, 'chair': 94, 'bicycle': 68, 'handbag': 64, 'backpack': 62, 'stop sign': 60, 'skateboard': 46}. Weather: {'WetNoon': 4155, 'ClearSunset': 2386, 'WetCloudyNoon': 2092, 'ClearNoon': 1482, 'HardRainNoon': 705}; town: {'Town04': 3904, 'Town03': 3693, 'Town02': 3223}. Silence runs by length (s): {'1': 640, '10+': 15, '2-3': 797, '4-6': 577, '7-9': 368}. First-frame reason: {'remain_silent': 14, 'direction_change': 2, 'obstacle_in_front': 2, 'enter_junction': 1, 'remain_silent_in_junction': 1}. Reason / instruction inconsistencies: none. `history[-1]` equals the previous frame's instruction in 1.0 of frames.

### real

| field | type | example |
|---|---|---|
| `id` | int | `15` |
| `video_id` | int | `2` |
| `video` | str | `"output/real_test/image_video_US_00/frame_275.mp4"` |
| `conversations` | list | `[{"from": "human", "value": "You are guiding a blind person. The blind person needs to approach the goal: [x,y...` |
| `QA` | list | `[{"q": "You are guiding a blind person. The blind person needs to approach the goal: [x,y]=[0.0, 1.0], which t...` |
| `goal_coord` | list | `[0.0, 1.0]` |
| `is_junction` | bool | `false` |
| `action` | str | `"walk straight"` |
| `image` | str | `"output/real_test/image_video_US_00/frame_275.jpg"` |
| `meta` | dict | `{"weather": "real", "town": "real"}` |
| `prediction` | str | `"None"` |
| `spoken_reason` | str | `"remain_silent"` |
| `seen` | bool | `true` |

Actions: {'walk straight': 985, 'turn slightly left': 29, 'turn slightly right': 17, 'stop': 17, 'turn left': 10, 'turn right': 1}. Conveyed obstacles: {} (directions {}). Scene objects (`description_cluster`): {}. Detector labels (`object_detection`): {}. Weather: {'real': 1059}; town: {'real': 1059}. Silence runs by length (s): {'1': 59, '2-3': 115, '4-6': 85, '7-9': 5}. First-frame reason: {'remain_silent': 7, 'constant_instruction': 2, 'remain_silent_in_junction': 1, 'obstacle_in_front': 1}. Reason / instruction inconsistencies: {'remain_silent but instruction given': 121, 'constant_instruction but no instruction': 19, 'obstacle_in_front but no instruction': 45, 'enter_junction but no instruction': 2, 'remain_silent_in_junction but instruction given': 5, 'None but no instruction': 1, 'exit_junction but no instruction': 1}. `history[-1]` equals the previous frame's instruction in None of frames.


# ORBIT: personal-object videos recorded by blind and low-vision people

Status as of 2026-10-08: **not downloaded**. The smallest benchmark variant (`validation_224.zip`, 4.15 GB) is over the
3 GB cap set for this check, so `data/orbit/` was not created. Everything below comes from the GitHub repo, the figshare
record and the dataset readme; sizes are the figshare file sizes (decimal GB).

## What it is

[ORBIT](https://github.com/microsoft/ORBIT-Dataset) (Massiceti et al., *ORBIT: A Real-World Few-Shot Dataset for Teachable
Object Recognition*, ICCV 2021, arXiv 2104.03841) is a collection of short phone videos of **personal objects** recorded by
people who are blind or low vision, built for *teachable* object recognisers: a user films an object a few times to teach it,
then the recogniser has to find that object in the user's cluttered surroundings.

* Unfiltered: 4,733 videos (3.16 M frames, 97 GB) of 588 objects from 97 collectors.
* Benchmark: 3,822 videos (2.69 M frames at 1080x1080, 83 GB) of 386 objects from 67 collectors, split by collector into
  train (44), validation (6) and test (17) users (`data/orbit_benchmark_mode_splits.json`). The figshare description quotes
  486 objects and 77 collectors for the benchmark; the repo readme and the split file say 386 and 67.
* Two video types per object: **clean** (the object alone, the support set a model personalises on) and **clutter** (the
  object somewhere in a realistic scene, the query set); the unfiltered set also has *clutter-pan* videos.
* Frames are stored as JPEGs in a `mode / collector / object / video type / video / frame` hierarchy, at 1080x1080 or
  pre-resized to 224x224.
* Extra annotations (`data/orbit_extra_annotations.zip` in the repo, 14.0 MB): one bounding box per frame for every clutter
  video (coordinates in the 1080x1080 frame, `null` when the object is out of frame) and, for every clean video, seven
  per-frame **quality issues** labelled by a vendor: `object_not_present`, `framing`, `viewpoint`, `blur`, `occlusion`,
  `overexposed`, `underexposed`.
* Object vocabulary (`data/object_clusters_benchmark.txt`, ~130 clusters of user-named objects): keys, wallet, phone,
  remote, headphones, glasses, medication and inhalers, white and symbol canes, guide-dog harness and lead, braille
  readers, hearing aids, front door, post box, wheelie bin, bus stop sign, mugs, bottles, appliances, furniture.
* Benchmark task: personalise on a test user's clean videos (any method), then predict which of that user's objects is in
  each of 200 sampled frames per clutter video; metric is frame accuracy with a 95% CI over 50 tasks per test user. The
  released baselines (ProtoNets / CNAPs / finetuning with EfficientNet or ViT backbones, 224x224) reach 65 to 75% frame
  accuracy; checkpoints are downloadable from the README table.

## How it could serve SeeForMe

1. **FIND mode for personal objects.** P-CACS honours a FIND request through the `goal` event (`w_goal`, `goal_tags`),
   which presumes a detector label that matches the request. COCO-80 cannot name most of what the ORBIT users filmed
   (keys, wallet, medication, remote, white cane, guide-dog harness, hearing aid, front door). ORBIT is the natural test bed
   for a "teach SeeForMe your object" path: the clean videos are the support set, the clutter videos plus their boxes are
   a realistic query set recorded by the same population, and the published protocol gives a comparable number. It also
   answers a design question directly: how many support frames a blind user must film before the object is found reliably.
2. **Blind-user image quality.** The seven per-frame quality issues on the clean videos are labelled examples of what
   blind users' phone footage actually looks like (framing, viewpoint, blur, occlusion, exposure). They can calibrate
   `seeforme/perception/preprocess.py` (`assess_quality`, `enhance_dark`) and the simulator's capture modifiers
   (`motion_blur`, `lighting`, the `poor_capture` scenario kind) against measured rates instead of guessed ones, and give a
   quick check of how YOLO11n confidence degrades with each issue.
3. **Personal-object detection targets.** The clutter-video boxes (one target object per frame, out-of-frame marked)
   are a small detection / tracking set for objects held close to a moving phone, useful for the tracker's hand-held jitter
   settings.
4. **What it is not.** Indoor, close-range, hand-held object videos: no walking, no hazards, no street scenes, no
   timing of when information was needed. It says nothing about *when to speak*; it is a perception and personalisation
   resource only.

## Licence

* Dataset: **CC BY 4.0** (figshare record, City, University of London, DOI 10.25383/city.14294597.v3). Cite
  Massiceti et al., ICCV 2021 (`@inproceedings{massiceti2021orbit}` in the repo README); the LITE training code is
  Bronskill et al., NeurIPS 2021.
* Code (`microsoft/ORBIT-Dataset`): MIT.
* Benchmark rules: validation and test users' data may not be used for training; personalisation may use only the test
  user's clean videos and labels (no boxes, no quality labels, no other users).

## Exact download commands and sizes

All archives are on figshare (`https://city.figshare.com/articles/dataset/_/14294597`, 148.95 GB in total); the repo's
`scripts/download_benchmark_dataset.sh <dir> 224` fetches the three 224x224 benchmark zips (52.4 GB) and unzips them into
`<dir>/orbit_benchmark_224/{train,validation,test}`; with any other frame size it fetches the full-size zips (82.2 GB) and
resizes. Per file:

| file | size | content | URL |
|---|---|---|---|
| `validation_224.zip` | 4.15 GB | benchmark, 6 validation users, 224x224 (smallest variant) | https://ndownloader.figshare.com/files/28368351 |
| `test_224.zip` | 16.43 GB | benchmark, 17 test users, 224x224 | https://ndownloader.figshare.com/files/28368072 |
| `train_224.zip` | 31.81 GB | benchmark, 44 train users, 224x224 | https://ndownloader.figshare.com/files/28368339 |
| `validation.zip` | 6.21 GB | benchmark, validation users, 1080x1080 | https://ndownloader.figshare.com/files/27188237 |
| `test.zip` | 26.14 GB | benchmark, test users, 1080x1080 | https://ndownloader.figshare.com/files/27346766 |
| `train.zip` | 49.88 GB | benchmark, train users, 1080x1080 | https://ndownloader.figshare.com/files/27189155 |
| `other.zip` | 14.32 GB | collectors / objects / videos outside the benchmark (unfiltered set only) | https://ndownloader.figshare.com/files/27380768 |
| `orbit_extra_annotations.zip` | 14.0 MB | per-frame boxes (clutter) and quality issues (clean) | https://github.com/microsoft/ORBIT-Dataset/raw/master/data/orbit_extra_annotations.zip |

The smallest useful download for the two SeeForMe uses above is the validation split at 224x224 plus the annotations
(about 4.2 GB on disk after the zip is removed; keep roughly 9 GB free while unzipping):

```bash
mkdir -p data/orbit/orbit_benchmark_224 data/orbit/orbit_benchmark_224/annotations
curl -L -o data/orbit/validation_224.zip https://ndownloader.figshare.com/files/28368351          # 4.15 GB
unzip -q data/orbit/validation_224.zip -d data/orbit/orbit_benchmark_224 && rm data/orbit/validation_224.zip
curl -L -o data/orbit/orbit_extra_annotations.zip https://github.com/microsoft/ORBIT-Dataset/raw/master/data/orbit_extra_annotations.zip
unzip -q data/orbit/orbit_extra_annotations.zip -d data/orbit/orbit_benchmark_224/annotations
git clone --depth 1 https://github.com/microsoft/ORBIT-Dataset external/ORBIT-Dataset   # loaders, summarize_dataset.py, baselines
python external/ORBIT-Dataset/scripts/summarize_dataset.py --data_path data/orbit/orbit_benchmark_224   # needs its own requirements
```

The repo's loaders target PyTorch 1.13 / Python 3.7 (`environment.yml`); for SeeForMe only the frame hierarchy and the
annotation JSONs are needed, which plain `pathlib` and `json` read.

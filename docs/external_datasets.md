# External datasets for SeeForMe (October 2026)

**None of these datasets provides ground-truth SILENCE / SPEAK / INTERRUPT / HAPTIC decisions.** Never treat object boxes or VideoQA labels as verified human scheduling targets.

| Dataset | Size and purpose | Access |
| --- | --- | --- |
| [EgoBlind](https://github.com/doc-doc/EgoBlind) | 1,392 egocentric BLV videos, 5,311 questions; task relevance and human requests | Author's Google Drive train/test links in README; research-use terms |
| [JRDB](https://jrdb.erc.monash.edu/dataset/) | 54 sequences, 60k frames, 2D/3D annotations; crowds, occlusion and tracking | Official JRDB downloads |
| [Aria Everyday Activities](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset/aea_download_dataset) | 143 recordings, roughly 7.5 hours, full data about 353 GB; sensor geometry / egomotion | Accept license, request expiring link JSON, then aria_dataset_downloader |
| [Ego4D](https://ego4d-data.org/docs/start-here/) | 3,600+ hours; out-of-distribution visual coverage | Accept license, get AWS credentials, install ego4d CLI; choose specific video UIDs |
| [TIMELI](https://timeli-icra.github.io/) | Reports 10,000 annotated walking clips for evaluation; closest timing baseline | Check release status; do not assume full annotations are downloadable |

## Acquisition order

1. Download permitted EgoBlind **training** data and a selected small sample of JRDB for crowded pedestrian cases; reserve existing public test data for evaluation.
2. Obtain several selected Aria sequences for depth/pose cross-checks. Download only required RGB/trajectory products, not the entire 353 GB.
3. After Ego4D access approval, use `pip install ego4d`; download metadata and annotations with `ego4d --output_directory data/external/ego4d --datasets annotations --metadata`, then choose video IDs. Avoid whole-dataset downloads.
4. Run `python scripts/dataset_inventory.py data/external --out data/external_inventory.jsonl` to catalog available videos.
5. Use `scripts/collect_real_video.py`, independent human annotations, and `scripts/export_real_sequences.py` to produce GRU-ready feature tensors.

## Training versus evaluation

- **Supervised imitation:** independently reviewed event windows, message slots and actions; oracle-only targets remain weak/synthetic.
- **DAgger:** perform interactive simulator rollouts and label policy-visited states with your reference/expert. Passive videos alone are not DAgger.
- **REINFORCE:** use closed-loop CampusBench-style simulation with realistic feedback. Passive videos do not reveal counterfactual action consequences.
- **Generalization:** split by original video, site, recording session and person (where applicable), never by adjacent frames or derived GRU windows.
- **Stress test:** create simultaneous competing events and simulated busy-channel states, because video alone does not imply INTERRUPT/HAPTIC behavior.

## Provenance

Record dataset, original source ID, author/license/attribution, clip ID, device and site, consent/access limits, split and annotation provenance. Keep raw video, credentials and any private BLV participant information out of the repository. Derive labels only after human verification. Suggested initial goal: 100–300 human-reviewed events with difficult negatives and uncertain cases.

# M2 plan: long-term appearance memory on GPU

Owner request (2026-11-12): "use GPU to recognize things outside window memory so it still remembers".

The tracker's ledger (`memory_ttl_s`, about 3 s) handles only short occlusions. M2 adds memory that lasts minutes:
- Every detection gets a CLIP ViT-B/32 embedding, computed on the GPU (MPS).
- When a track dies, its label, embedding and announced state go into a re-identification gallery.
- A new track of the same label whose embedding has cosine ≥ θ with a gallery entry inherits that announced state.
- The scheduler then treats it as known, not new.

| Slice | Deliverable | Status |
|---|---|---|
| S1 | `seeforme/tracking/reid.py` gallery; `TrackerConfig.reid_horizon_s` (0 = off) and `reid_threshold`; `Detection.emb` and `Track.emb`/`reid_of` | done, tests in `tests/test_reid_memory.py` |
| S2 | CLIP encoder and side-car caches in `data/egoblind_app/` | done: 592 videos, 414k detections |
| S3 | Evaluation: P-CACS, IL and the online learner with re-ID on and off on the dev stream | done, `docs/memory.md`; hand check of 50 matches pending |
| S4 | Choose θ and the horizon on `select` only, then a single `--final` run | todo |

S3 measures three things:
- the repeat-cue rate (cues about a re-identified object);
- precision;
- coverable recall, which must not drop.

Acceptance: fewer repeat cues, coverable recall unchanged (95% CI of the difference includes 0 or is positive), and re-ID precision checked by hand on 50 sampled matches.

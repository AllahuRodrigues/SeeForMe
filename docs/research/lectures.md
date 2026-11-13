# Course lectures mapped to SeeForMe (2026-11-12)

Two lecture sets, read page by page; page numbers are PDF page indices. "Derived" marks reasoning built from slide material rather than slide content.

- **Course A:** Yale CPSC 480/580, *Introduction to Computer Vision* (Alex Wong), lectures 1 to 14.
- **Course B:** *Statistical Learning Theory* (Omar Montasser, Yale), lectures 1 to 10. Online learning appears in L9 (mistake bounds, Halving, Littlestone dimension, online-to-batch) and L10 (agnostic regret, Multiplicative Weights, Littlestone-class regret, boosting from Multiplicative Weights); bandits, online convex optimisation and delayed feedback are not covered. L10 mapping: `docs/research/benchmarks.md`.
- Not covered by course A: Hough transform, optical-flow algorithms, epipolar geometry, tracking, time to contact.

## Concept maps

### Course A
| Lecture | Key methods |
|---|---|
| L01 | Recognition, reconstruction, registration; detection, segmentation, SLAM |
| L02 | Homogeneous points and lines; intersection by cross product; parallel lines meet at infinity; transform hierarchy up to homography (8 DoF); lines map by H^-T |
| L03 | Pinhole; perspective x = f X / Z; magnification m = abs(f / Z), inversely proportional to depth (p14); vanishing point (f dx/dz, f dy/dz) (p20); intrinsics K |
| L04 | Convolution, padding, box and Gaussian kernels, separability, median filter |
| L05 | Gradients; Sobel, Prewitt; Laplacian, LoG; Canny (smooth, gradient, NMS, hysteresis) |
| L06 | Moravec; structure tensor; Harris R = det S - k tr(S)^2; Shi-Tomasi |
| L07 | Normalized LoG scale space; Gaussian and Laplacian pyramids |
| L08 | HOG (9 unsigned bins, block L2 normalization); DoG; SIFT 128-D; matching by distance or NCC |
| L09 | Registration as camera motion; least-squares translation and affine; homography from 4 points (smallest eigenvector); valid for pure rotation or planar / far scenes (p20) |
| L10 | Forward and inverse warping; bilinear interpolation |
| L11 | RANSAC; T >= log(1-p) / log(1-(1-e)^s); s=4, e=50%, p=0.99 gives 72 iterations; inlier threshold from noise |
| L12 | MLPs, CNNs, pooling; encoder-decoder for segmentation, depth, flow; attention and ViT |
| L13 | Template matching; PCA; bag of visual words; AlexNet, VGG, ResNet |
| L14 | Integral images, Viola-Jones cascade; Dalal-Triggs HOG + linear SVM + NMS; DPM; R-CNN; YOLO |

### Course B
| Lecture | Key methods |
|---|---|
| L1 | PAC; finite-class bound; Hoeffding; uniform convergence; test-set bound L_D(h) <= L_S~(h) + sqrt(ln(1/delta) / 2m~) (p27) |
| L2-L3 | Growth function, VC dimension, Sauer-Shelah; statistical no free lunch (L3 p15) |
| L4 | Efficient PAC; hardness; surrogate losses (logistic, hinge) (p32) |
| L5 | MDL, SRM as a regularization path (p11-17); PAC-Bayes with KL(Q, P) and Gibbs posterior (p20-22) |
| L6-L7 | AdaBoost and its training-error bound; margins bound; sample compression |
| L8 | Rademacher complexity; linear predictors: L2 ball scales with sqrt(avg norm(x)^2 / m), L1 ball with sqrt(2 log(2d) / m) (p13-14) |
| L10 | Agnostic regret vs best fixed expert (p7-8); deterministic learners suffer T/2 regret (p9); Multiplicative Weights, regret <= sqrt((T/2) ln\|H\|) (p11-15); Littlestone-class regret (p16-19); boosting via Multiplicative Weights (p20-22) |
| L9 | Online protocol and no free lunch; mistake bounds; Halving <= log2 abs(H); conservative algorithms (p9); thresholds have infinite Littlestone dimension (p10-11); online-to-batch "longest running survivor" (p18) |

## Ranked ideas for SeeForMe

Every anticipation idea goes in at decision level in `OnlinePolicy.decide` and must pass `scripts/feature_auc.py` (+0.02 finetune AUC, interval above 0) before a 3-seed online run, with settings chosen on `select`.

1. **Looming time to contact from box height** (A L3 p13-14; derived tau = h / (dh/dt) = Z / closing speed). `Track.looming_ttc()`. **Tested 2026-11-12: fails the gate (+0.009 [-0.004, +0.023]); see `docs/online.md` item 9.**
2. **Camera ego-motion: Shi-Tomasi corners, descriptor matching, RANSAC similarity / homography** (A L6 p21-27, L8 p25-27, L9 p2-3, p13-29, L11 p7-12). Features: yaw rate and its change (turns), a "camera still" flag (the user stopped), residual parallax (an image-based walking-speed proxy that exists without ranged tracks), RANSAC inlier ratio (covered lens, wall, sudden scene change). New `seeforme/perception/egomotion.py`, one decode pass at 5 fps. Highest expected value: stopping and turning are when blind pedestrians ask, and it works when the detector sees nothing.
3. **Ground-region oriented-gradient "stripes ahead" score for stairs, crosswalks, curbs** (A L5 p9, p23-29; L8 p7-11; L3 p20). HOG energy in near-horizontal bins below the horizon, times a regularity term of horizontal-edge rows (derived). Also usable as a pseudo-candidate when the detector is silent. Cheap test: AUC on ~50 frames before stairs / crossing / curb questions vs ~150 random frames (pass >= 0.75).
4. **Sequential RANSAC lines, vanishing point, per-frame pitch and path heading, curb lines** (A L11 p4-9, L2 p6-9, L3 p20, p29). Replaces the fixed 15 degree pitch in `CameraModel`, which feeds distance, lateral offset, path probability and TTC.
5. **Focus of expansion from the matches in idea 2** (derived from A L3 p20 and L2 p7). Heading and per-track angular distance to the FOE as a depth-free collision cue. Only if idea 2 passes.
6. **Gaussian-pyramid scene-change signal** (A L7 p17-20). Distance between coarse frames now and 1 s / 3 s ago: doorways, intersections, lighting changes. Nearly free in the same decode pass.
7. **Adding features without overfitting 160 events: L1-constrained logistic and SRM over nested feature groups** (B L8 p13-14, L5 p11-17, L4 p32). The L1-ball complexity grows only with log d.
8. **HOG + linear classifier for doors, boosted cascade for stairs** (A L14 p14-28, B L6, L7). Ranked last: labels are scarce and a CNN fine-tune usually wins.

Run order: 1 (done), then one decode pass computing 2, 6 and 3, then 4, then 5; 7 throughout; 8 only if doors remain the main miss.

## What course B says about the thesis formulation

1. **Anticipation needs prior structure** (B L9 p4-5; L3 p15). With no assumed class an adversary forces a mistake every round; prior knowledge enters through the hypothesis class. This is the formal form of "features, not the feedback channel, are the bottleneck". Decisions within a video are correlated, so the effective sample size is closer to the ~160 events than to the ~20,800 candidate rows (derived).
2. **Why one-sided feedback cost nothing here** (B L9 p9, interpretation). A conservative learner that updates only on mistakes keeps its mistake bound. With p_ask = 1 every silent miss becomes a question and speaking reveals its own outcome, so on the scored region the learner sees nearly all its mistakes. Consistent with 0.204 vs 0.200; the slide guarantees assume immediate labels, and the delay curve is the empirical counterpart.
3. **No worst-case online guarantee for a threshold policy** (B L9 p10-11, p17). Thresholds and halfspaces have infinite Littlestone dimension. The thesis should use a stochastic / agnostic framing or a margin assumption (B L7 p8-12), not mistake bounds.
4. **Freezing the online learner** (B L9 p18). "Longest running survivor" online-to-batch conversion gives one fixed policy with error <= epsilon w.p. 1 - delta after O(M log(M/delta) / epsilon) examples: a principled "learn on dev, freeze for the user study" protocol.
5. **Warm start as a PAC-Bayes effect** (B L5 p20-22, interpretation). Carrying the simulator's full state passes a narrow prior; moving to real video costs a large KL, so the learner stays near the simulator (-0.058). Weights only keeps the mean with a broad prior, and the harm disappears. Transfer the simulator as a broad prior, not as a posterior.
6. **How large a `select` difference can be trusted** (B L1 p27). With 11 coverable events (seeds reuse them), sqrt(ln 20 / 22) = 0.37; on dev with 160 events, 0.097. `select` gaps below about 0.3 cannot be certified by this bound: `select` is indicative only, and more labelled events are needed before final claims.
7. **Model capacity** (B L4 p32-33, L8 p13-14, L7). Keep the logistic surrogate and grow features under an L1 constraint; the margins bound explains why gradient boosting overfit with few events.

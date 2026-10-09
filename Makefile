PY := .venv/bin/python

.PHONY: egoblind-holdout egoblind-tune setup test scenarios models tune experiments ablation plots latency all demo clean campus tune-pcacs scheduler campus-experiments campus-report web serve serve-ssl live egoblind-download egoblind-cache egoblind-scenarios egoblind-experiments egoblind-scheduler egoblind-all campus-v2 campus-v2-experiments campus-v2-perfect memory-grid stress archs sign-taxonomy guidedog inventory datasets-studies coverage egoblind-diagnose egoblind-cache-ow egoblind-thr export-models live-elevenlabs smoke repro-check manifests latency-pcacs decompose

setup:
	python3 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q tests

scenarios:
	$(PY) scripts/make_scenarios.py

models:
	$(PY) scripts/train_text_classifier.py
	$(PY) scripts/train_scene_context.py

tune:
	$(PY) scripts/tune_cacs.py --trials 150

deep:
	$(PY) scripts/train_deep_context.py

export-models:
	$(PY) scripts/export_models.py

priority:
	$(PY) scripts/train_priority_model.py

experiments:
	$(PY) scripts/run_experiments.py

ablation:
	$(PY) scripts/run_ablation.py

report:
	$(PY) scripts/write_results_md.py

plots:
	$(PY) scripts/make_plots.py

latency:
	$(PY) scripts/benchmark_latency.py

latency-pcacs:
	$(PY) scripts/benchmark_latency_pcacs.py

all: scenarios models tune priority deep experiments ablation plots report

demo:
	$(PY) scripts/demo_sim.py --kind hallway --policy confidence_threshold
	$(PY) scripts/demo_sim.py --kind hallway --policy cacs

live:
	$(PY) scripts/run_live.py --policy pcacs --speak

# needs ELEVENLABS_API_KEY in the environment or .env (falls back to the default speaker without one)
live-elevenlabs:
	$(PY) scripts/run_live.py --policy pcacs --speak --elevenlabs

# ---- CampusBench / P-CACS -------------------------------------------------
campus:
	$(PY) scripts/make_campus.py

tune-pcacs:
	$(PY) scripts/tune_pcacs.py --trials 150
	$(PY) scripts/export_web_config.py

scheduler:
	$(PY) scripts/train_scheduler.py

campus-experiments:
	$(PY) scripts/run_campus.py --split test --policies oracle,confidence_threshold,confidence_cooldown,cacs,pcacs,pcacs-no_ttc,pcacs-no_depth,pcacs-no_budget,pcacs-no_interrupt,pcacs-no_queue,pcacs-no_haptic,pcacs-no_slots,pcacs-quiet,pcacs-detailed,pcacs_learned:il,pcacs_learned:rl

campus-report:
	$(PY) scripts/make_campus_plots.py
	$(PY) scripts/write_campus_results.py

campus-all: campus tune-pcacs scheduler campus-experiments campus-report

# ---- EgoBlind (real BLV egocentric video, weak labels from the users' questions) --------
egoblind-download:
	$(PY) scripts/download_egoblind.py --all --retry-every 20

egoblind-cache:
	$(PY) scripts/cache_egoblind_perception.py --splits holdout,select,finetune,train,test

egoblind-scenarios:
	$(PY) scripts/build_egoblind_scenarios.py

# 70 / 15 / 5 / 10 by source recording (scripts/egoblind_common.py): train 70, finetune 15 + select 5, holdout 10.
# The holdout is never used for training or selection; EgoBlind's official test is the external benchmark.
EGO_POLICIES = oracle,confidence_threshold,confidence_cooldown,cacs,pcacs,pcacs_egoblind,pcacs_learned:il,pcacs_learned:rl,egoblind_learned:il,egoblind_learned:rl,egoblind_learned:il@models/egoblind_only,egoblind_learned:rl@models/egoblind_only

egoblind-scheduler:
	$(PY) scripts/train_scheduler_egoblind.py --train egoblind --dagger-iters 1 --rl-epochs 6 --out models/egoblind_only
	$(PY) scripts/train_scheduler_egoblind.py --train egoblind,campus --dagger-iters 1 --rl-epochs 6 --out models/egoblind

egoblind-holdout:
	$(PY) scripts/run_egoblind.py --split holdout --policies $(EGO_POLICIES) --tag final
	$(PY) scripts/run_egoblind.py --split holdout --policies $(EGO_POLICIES) --noise p_miss=0.2,depth_rel=0.15,p_blackout=0.05 --tag degraded

egoblind-experiments:
	$(PY) scripts/run_egoblind.py --split test --policies $(EGO_POLICIES) --tag official
	$(PY) scripts/run_egoblind.py --split test --policies $(EGO_POLICIES) --noise p_miss=0.2,depth_rel=0.15,p_blackout=0.05 --tag official_degraded

egoblind-diagnose:
	$(PY) scripts/diagnose_egoblind_recall.py --split holdout
	$(PY) scripts/diagnose_egoblind_recall.py --split train --policies pcacs,pcacs_learned:rl,egoblind_learned:il

# linear P-CACS tuned by random search on EgoBlind `train` replays -> configs/pcacs_egoblind.yaml (policy pcacs_egoblind)
egoblind-tune:
	$(PY) scripts/tune_pcacs.py --domain egoblind --trials 80

egoblind-thr:
	$(PY) scripts/run_egoblind.py --split train --policies pcacs,pcacs-thr0.6,pcacs-thr0.3,pcacs-thr0.0 --tag thr

# open-vocabulary perception (YOLO-World, perception.NAV_VOCAB) in its own cache directory
egoblind-cache-ow:
	SEEFORME_EGOBLIND_CACHE=data/egoblind/cache_ow $(PY) scripts/cache_egoblind_perception.py --splits holdout,select,finetune,train --detector yolo-world
	SEEFORME_EGOBLIND_CACHE=data/egoblind/cache_ow $(PY) scripts/run_egoblind.py --split holdout --tag ow
	SEEFORME_EGOBLIND_CACHE=data/egoblind/cache_ow $(PY) scripts/diagnose_egoblind_recall.py --split holdout --tag ow

# scenarios are built twice on purpose: once from the weak labels, then again after caching to refine with perception.
# Separate $(MAKE) calls because make runs a prerequisite only once per invocation.
egoblind-all:
	$(MAKE) egoblind-scenarios
	$(MAKE) egoblind-cache
	$(MAKE) egoblind-scenarios
	$(MAKE) egoblind-scheduler
	$(MAKE) egoblind-tune
	$(MAKE) egoblind-holdout
	$(MAKE) egoblind-experiments

# ---- CampusBench v2 (research/ blueprint: contention, occlusion + memory, multilingual wayfinding, OOD) ----
campus-v2:
	$(PY) scripts/make_campus.py --out data/campus_v2 --kinds all

campus-v2-experiments:
	SEEFORME_CAMPUS=v2 $(PY) scripts/run_campus.py --split test --repeats 1 --tag zeroshot
	SEEFORME_CAMPUS=v2 $(PY) scripts/run_campus.py --split test --repeats 1 --policies oracle,pcacs,pcacs_learned:il,pcacs_learned:rl --memory-ttl 6 --tag mem

campus-v2-perfect:
	SEEFORME_CAMPUS=v2 $(PY) scripts/run_campus.py --split test --repeats 1 --policies oracle,confidence_cooldown,pcacs,pcacs_learned:il,pcacs_learned:rl --perfect-perception --tag perfect

memory-grid:
	SEEFORME_CAMPUS=v2 $(PY) scripts/memory_grid.py

stress:
	SEEFORME_CAMPUS=v2 $(PY) scripts/run_stress_matrix.py --sample 200 --tag s200

archs:
	for a in mlp1 mlpk attn gru; do SEEFORME_CAMPUS=v2 $(PY) scripts/train_scheduler.py --arch $$a --dagger-iters 1 --epochs 15 --rl-epochs 0 --out models/arch_$$a; done
	SEEFORME_CAMPUS=v2 $(PY) scripts/train_scheduler.py --arch gru --memory-ttl 6 --dagger-iters 1 --epochs 15 --rl-epochs 0 --out models/arch_gru_mem
	SEEFORME_CAMPUS=v2 $(PY) scripts/compare_archs.py

sign-taxonomy:
	$(PY) scripts/eval_sign_taxonomy.py

guidedog:
	$(PY) scripts/download_guidedog.py && $(PY) scripts/guidedog_eval.py

inventory:
	$(PY) scripts/dataset_inventory.py

# external dataset studies (see docs/datasets.md)
datasets-studies:
	$(PY) scripts/blvnav_taxonomy.py
	$(PY) scripts/mendeley_navassist.py
	$(PY) scripts/timeli_inspect.py && $(PY) scripts/timeli_to_events.py
	$(PY) scripts/vizwiz_robustness.py

# ~3 min: unit tests, GuideDog synthetic rows, a 3-video EgoBlind replay. Run before any long job.
smoke:
	$(PY) -m pytest -q tests
	$(PY) scripts/guidedog_eval.py --smoke
	$(PY) scripts/run_egoblind.py --split select --limit 3 --policies oracle,pcacs --tag smoke

# re-runs committed CampusBench v2 / EgoBlind results and checks they match (results/repro_check.md)
repro-check:
	$(PY) scripts/repro_check.py

# dataset / split / model provenance (data/manifests, data/splits, models/*/PROVENANCE.json)
manifests:
	$(PY) scripts/build_manifests.py

decompose:
	$(PY) scripts/decompose_pcacs_utility.py --split train
	$(PY) scripts/decompose_pcacs_utility.py --split holdout

coverage:
	cd research && ../$(PY) sample_coverage.py --per-case 8 && ../$(PY) -m pytest -q test_catalog.py

# ---- web app + server ---------------------------------------------------------
web:
	cd web && npm install && npm run build

serve:
	$(PY) server/app.py --warm

serve-ssl:
	$(PY) server/app.py --warm --ssl

# removes only what `make all` regenerates; later-stage results (latency, export, sign taxonomy, CampusBench,
# EgoBlind, GuideDog, repro check) and models from other targets are kept
clean:
	rm -rf results/figures data/scenarios configs/cacs_tuned.yaml \
	  models/text_importance.joblib models/scene_context.joblib models/priority_*.joblib \
	  results/text_classifier_report.json results/scene_context_report.json results/tuning.json results/priority_model_report.json \
	  results/deep_context_report.json results/per_scenario_test.csv results/summary_test.json results/summary_by_kind_test.json \
	  results/summary_test.md results/ablation_test.json results/ablation_per_scenario_test.csv results/ablation_test.md results/RESULTS.md

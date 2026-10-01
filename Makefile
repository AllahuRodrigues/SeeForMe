PY := .venv/bin/python

.PHONY: setup test scenarios models tune experiments ablation plots latency all demo clean campus tune-pcacs scheduler campus-experiments campus-report web serve serve-ssl live

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

export:
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

all: scenarios models tune priority deep experiments ablation plots report

demo:
	$(PY) scripts/demo_sim.py --kind hallway --policy confidence_threshold
	$(PY) scripts/demo_sim.py --kind hallway --policy cacs

live:
	$(PY) scripts/run_live.py --policy pcacs --speak

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

# ---- web app + server ---------------------------------------------------------
web:
	cd web && npm install && npm run build

serve:
	$(PY) server/app.py --warm

serve-ssl:
	$(PY) server/app.py --warm --ssl

clean:
	rm -rf results/*.csv results/*.json results/*.md results/figures data/scenarios models/*.joblib configs/cacs_tuned.yaml

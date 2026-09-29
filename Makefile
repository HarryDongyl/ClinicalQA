UV ?= uv
PY := $(UV) run --frozen
RUN ?= q5filtered_lr5e5
RUNS ?= raw_lr1e4 raw_lr5e5 q5filtered_lr5e5
SPLIT ?= val
ADAPTER ?=
LABEL ?= $(RUN)
TRAIN_CONFIG := configs/train/$(RUN).yaml
RESUME ?=

.PHONY: help setup setup-cpu lock data analyze views test all format audit-masks smoke train eval epochs \
        compare core final-eval clean data-check preflight gpu-smoke freeze

help:
	@echo "setup        install the locked environment incl. the train extra (uv sync --frozen --extra train)"
	@echo "setup-cpu    analysis-only environment (no torch)"
	@echo "lock         re-resolve dependencies and regenerate requirements.txt"
	@echo "data         copy + validate provided data into data/ (writes data/MANIFEST.json)"
	@echo "analyze      data statistics + quality checks -> reports/"
	@echo "views        raw and Q5-filtered train selections"
	@echo "format       native Qwen chat conversations -> data/sft/, token lengths -> reports/token_lengths.json"
	@echo "audit-masks  loss-mask audit + formatted examples -> reports/"
	@echo "smoke        tiny overfit + adapter reload + rollout with Qwen3-0.6B (not a result)"
	@echo "train        RUN=raw_lr1e4|raw_lr5e5|q5filtered_lr5e5 [RESUME=latest]"
	@echo "eval         RUN=base|train_run [ADAPTER=path LABEL=name] rollout + score on val"
	@echo "epochs       RUN=train_run  evaluate epoch checkpoints on val and select"
	@echo "compare      A=label B=label  paired val comparison"
	@echo "core         train/evaluate the current RUN after GPU smoke has passed (val only)"
	@echo "data-check   verify training views, quarantine policy and unchanged labels"
	@echo "preflight    data checks plus CUDA/precision/token-prefix checks (RunPod)"
	@echo "gpu-smoke    4B NF4 overfit + reload + both tools, required before formal training"
	@echo "freeze       RUNS='...' select across validation checkpoints and write final config"
	@echo "final-eval   frozen test comparison from configs/final_eval.yaml (runs once)"
	@echo "test         unit tests"

setup:
	$(UV) sync --frozen --extra train

setup-cpu:
	$(UV) sync --frozen

lock:
	$(UV) lock
	$(UV) export --frozen --no-hashes --no-emit-project --extra train --format requirements-txt -o requirements.txt

data:
	$(PY) python -m clinqa.data_io --config configs/data.yaml

analyze: data
	$(PY) python -m clinqa.analyze --config configs/analysis.yaml

views: data
	$(PY) python -m clinqa.data_views --variant raw
	$(PY) python -m clinqa.data_views --variant q5_filtered

test:
	$(PY) pytest

all: data analyze views test

format: views
	$(PY) python -m clinqa.formatting --config configs/format_core.yaml

audit-masks: format
	$(PY) python -m clinqa.audit_masks --config configs/format_core.yaml
	$(PY) python -m clinqa.audit_masks --config configs/format_core.yaml --view q5_filtered --output-dir reports/q5filtered

smoke: views
	$(PY) python -m clinqa.smoke --config configs/smoke.yaml

data-check: views
	$(PY) python -m clinqa.preflight

preflight: data-check
	$(PY) python -m clinqa.preflight --gpu

gpu-smoke:
	$(PY) python -m clinqa.smoke --config configs/train/gpu_smoke.yaml $(if $(RESUME),--resume $(RESUME),)

train:
	@test -f $(TRAIN_CONFIG) || (echo "Unknown run $(RUN); see make help"; exit 1)
	$(PY) python -m clinqa.train --config $(TRAIN_CONFIG) $(if $(RESUME),--resume $(RESUME),)

eval:
	$(PY) python -m clinqa.evaluate generate --run $(RUN) --split $(SPLIT) --label $(LABEL) \
		$(if $(ADAPTER),--adapter $(ADAPTER),)

epochs:
	$(PY) python -m clinqa.evaluate epochs --run $(RUN)

compare:
	$(PY) python -m clinqa.evaluate compare --a $(A) --b $(B) --split val

core:
	$(MAKE) train RUN=$(RUN)
	$(MAKE) eval RUN=base
	$(MAKE) epochs RUN=$(RUN)

freeze:
	$(PY) python -m clinqa.evaluate freeze --runs $(RUNS)

final-eval:
	$(PY) python -m clinqa.evaluate final --final-config configs/final_eval.yaml

clean:
	rm -rf data/processed/*/train.jsonl data/sft outputs/smoke .pytest_cache

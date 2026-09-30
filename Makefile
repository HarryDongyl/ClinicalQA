UV ?= uv
PY := $(UV) run --frozen
RUN ?= q5filtered_lr5e5
RUNS ?= raw_lr1e4 raw_lr5e5 q5filtered_lr5e5
SPLIT ?= val
ADAPTER ?=
LABEL ?= $(RUN)
TRAIN_CONFIG := configs/train/$(RUN).yaml
RESUME ?=
EVAL_CONFIG ?= configs/eval_core.yaml

.PHONY: help setup setup-cpu lock data analyze views test all format audit-masks smoke train eval epochs \
        compare core final-eval clean data-check preflight gpu-smoke freeze score-v2 prompt-ablation \
        w2-round w2-epochs w2-stable score-v21 w2-score

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
	@echo "score-v2     validation-only contract scoring into a new report directory; no automatic selection"
	@echo "prompt-ablation  W2-E1 generation only: system v1 vs v2 on fixed wave-one weights (val, GPU, batch 4)"
	@echo "w2-round     RunPod: whole wave-two round (E1 + filtered LR ladder at mb1) [CROSS=1 adds prompt v2; MB=mb4]"
	@echo "w2-epochs    RUN=w2_run PROMPT=v1|v2  evaluate both epoch checkpoints on val, no selection (GPU)"
	@echo "w2-stable    RUN=w2_run  exit 1 on non-finite loss/grad_norm or unfinished training"
	@echo "score-v21    LABELS='...' OUT=dir  candidate scorer v2.1 on val labels (diagnostic, CPU)"
	@echo "w2-score     local CPU: v2.1 + bounded v2 + legacy paired comparisons + rollout diffs for wave two"

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

score-v2:
	$(PY) python scripts/rescore_validation_v2.py $(if $(OUT),--out $(OUT),)

# W2-E1 generation (GPU): arm x fixed wave-one weights under one code state, batch 4. Labels: w2p_<arm>_<run>.
# Scoring is separate and local (w2-score); the pod only generates and runs the legacy scorer (D-047).
PROMPT_ABLATION_RUNS := base raw_lr1e4 raw_lr5e5 q5filtered_lr5e5
prompt-ablation:
	@for arm in v1 v2; do for r in $(PROMPT_ABLATION_RUNS); do \
	  $(PY) python -m clinqa.evaluate --config configs/eval_w2_prompt_$$arm.yaml generate --run $$r --split val \
	    --label w2p_$${arm}_$$r || exit 1; done; done

w2-round:
	CROSS=$(CROSS) MB=$(MB) bash scripts/run_w2_round.sh

PROMPT ?= v1
w2-epochs:
	$(PY) python scripts/w2_epochs.py generate --run $(RUN) --prompt $(PROMPT)

w2-stable:
	$(PY) python scripts/w2_epochs.py stable --run $(RUN)

score-v21:
	@test -n "$(LABELS)" && test -n "$(OUT)" || (echo "usage: make score-v21 LABELS='a b' OUT=reports/new_dir"; exit 1)
	$(PY) python scripts/score_v21_val.py --out $(OUT) --labels $(LABELS)

# Wave-two scoring on CPU after the pod is stopped. Scores every wave-two validation label present:
# E1 (w2p_*) and the LR-ladder epoch checkpoints (w2_filtered_*_step*). No winner is declared (D-049).
W2_OUT = $(or $(OUT),reports/w2_round)
W2_LABELS = $(sort $(patsubst outputs/%/val/scored.jsonl,%,$(wildcard outputs/w2p_*/val/scored.jsonl outputs/w2_filtered_*_step*/val/scored.jsonl)))
w2-score:
	@test -n "$(W2_LABELS)" || (echo "no wave-two validation outputs found"; exit 1)
	$(PY) python scripts/score_v21_val.py --out $(W2_OUT)/v21 --labels $(W2_LABELS)
	$(PY) python scripts/rescore_validation_v2.py --out $(W2_OUT)/bounded_v2 --labels $(W2_LABELS)
	@for r in $(PROMPT_ABLATION_RUNS); do \
	  test -f outputs/w2p_v1_$$r/val/scored.jsonl && test -f outputs/w2p_v2_$$r/val/scored.jsonl || continue; \
	  $(PY) python -m clinqa.evaluate compare --a w2p_v1_$$r --b w2p_v2_$$r --split val >/dev/null && \
	  $(PY) python scripts/compare_rollouts.py w2p_v1_$$r w2p_v2_$$r --out $(W2_OUT)/rollout_diffs/v1_vs_v2_$$r.json \
	  || exit 1; done
	@for pair in base:base raw_lr1e4:raw_lr1e4_step000125 raw_lr5e5:raw_lr5e5_step000125 q5filtered_lr5e5:q5filtered_lr5e5_step000121; do \
	  r=$${pair%%:*}; w1=$${pair#*:}; test -f outputs/w2p_v1_$$r/val/scored.jsonl || continue; \
	  $(PY) python scripts/compare_rollouts.py $$w1 w2p_v1_$$r --out $(W2_OUT)/rollout_diffs/drift_$$r.json || exit 1; done
	@# References are the E1 v1 regenerations (same code state and generation batch), not the wave-one outputs.
	@for l in $(filter w2_filtered_%,$(W2_LABELS)); do for ref in w2p_v1_raw_lr1e4 w2p_v1_q5filtered_lr5e5; do \
	  test -f outputs/$$ref/val/scored.jsonl || continue; \
	  $(PY) python -m clinqa.evaluate compare --a $$ref --b $$l --split val >/dev/null || exit 1; done; done
	@echo "Reports in $(W2_OUT); legacy paired comparisons in outputs/compare_*_val_full.{md,json}"

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
	$(PY) python -m clinqa.evaluate --config $(EVAL_CONFIG) generate --run $(RUN) --split $(SPLIT) --label $(LABEL) \
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

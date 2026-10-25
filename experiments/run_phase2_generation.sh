#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

LOG_DIR="experiments/phase2_logs"
mkdir -p "$LOG_DIR"
trap 'echo "FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

if [[ $DRY_RUN -eq 1 ]]; then
  MODEL="dry-run-local-stub"
  TAG_SUFFIX="_dryrun_v2"
  echo "Phase 2 dry-run: local stub only; no API key is read and no network request is made."
else
  : "${DASHSCOPE_API_KEY:?DASHSCOPE_API_KEY is not set; export it in the launching terminal}"
  : "${QWEN_MODEL:?QWEN_MODEL is not set; export the exact DashScope model id}"
  MODEL="$QWEN_MODEL"
  TAG_SUFFIX=""
fi

ping_tag="phase2_ping${TAG_SUFFIX}"
ping_log="$LOG_DIR/ping${TAG_SUFFIX}.log"
ping_command=(
  python3 experiments/generate_rule_library.py
  --tag "$ping_tag" --model "$MODEL" --temperature 0.8
  --feature_set theory11_plus3 --scale M --difficulty service_heavy
  --enable_thinking false --ping
)
if [[ $DRY_RUN -eq 1 ]]; then
  ping_command+=(--dry_run)
fi
echo "START connectivity check: model=$MODEL"
"${ping_command[@]}" 2>&1 | tee "$ping_log"
echo "Connectivity check passed: model=$MODEL"

generation_complete() {
  local tag="$1"
  python3 - "$tag" <<'PY'
import json
import pathlib
import sys

tag = sys.argv[1]
state = pathlib.Path("experiments") / f"llm_candidates_{tag}.json"
library = pathlib.Path("experiments") / f"llm_library_{tag}_6.json"
if not state.exists() or not library.exists():
    raise SystemExit(1)
payload = json.loads(state.read_text(encoding="utf-8"))
raise SystemExit(0 if int(payload.get("completed_rounds", 0)) >= 4 else 1)
PY
}

for base_tag in p3b1 p3b2 p3b3; do
  tag="${base_tag}${TAG_SUFFIX}"
  log="$LOG_DIR/generate_${tag}.log"
  if generation_complete "$tag"; then
    echo "SKIP completed generation: $tag"
  else
    echo "START generation: $tag"
    generation_command=(
      python3 experiments/generate_rule_library.py
      --tag "$tag" --model "$MODEL" --temperature 0.8
      --rounds 4 --per_round 10 --library_size 6
      --feature_set theory11_plus3 --scale M
      --difficulty service_heavy --show_classic_reference
      --enable_thinking false
    )
    if [[ $DRY_RUN -eq 1 ]]; then
      generation_command+=(--dry_run)
    fi
    "${generation_command[@]}" 2>&1 | tee "$log"
    generation_complete "$tag" || {
      echo "Generation did not produce a complete four-round state: $tag" >&2
      exit 1
    }
  fi
done

for base_tag in p3b1 p3b2 p3b3; do
  tag="${base_tag}${TAG_SUFFIX}"
  for scale in S M L; do
    for variant in g mix; do
      output="experiments/llm_library_${tag}_${scale}_${variant}.json"
      report="experiments/gated_report_${tag}_${scale}_${variant}.md"
      log="$LOG_DIR/gate_${tag}_${scale}_${variant}.log"
      if [[ -s "$output" && -s "$report" ]]; then
        echo "SKIP completed gated library: ${tag}_${scale}_${variant}"
        continue
      fi
      gate_command=(
        python3 experiments/build_gated_library.py
        --candidates "experiments/llm_candidates_${tag}.json"
        --tag "${tag}_${scale}_${variant}" --K 6
        --sim_threshold 0.9 --gain_threshold 0.002
        --scale "$scale" --difficulty service_heavy
        --feature_set theory11_plus3 --gate_baseline best_classic
        --out "$output" --report "$report"
      )
      if [[ "$variant" == "mix" ]]; then
        gate_command+=(--extra_rules experiments/classic4_library.json)
      fi
      echo "START gated library: ${tag}_${scale}_${variant}"
      "${gate_command[@]}" 2>&1 | tee "$log"
    done
  done
done

report="experiments/pre_formal_check3${TAG_SUFFIX}.md"
if [[ -s "$report" ]]; then
  echo "SKIP completed validation report: $report"
else
  tags=("p3b1${TAG_SUFFIX}" "p3b2${TAG_SUFFIX}" "p3b3${TAG_SUFFIX}")
  echo "START validation report: $report"
  python3 experiments/pre_formal_check3.py \
    --tags "${tags[@]}" --scales S M L --difficulty service_heavy \
    --seed_start 100 --seed_end 149 --output "$report" \
    2>&1 | tee "$LOG_DIR/pre_formal_check3${TAG_SUFFIX}.log"
fi

echo "Phase 2 pipeline completed successfully: dry_run=$DRY_RUN"

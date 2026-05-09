#!/usr/bin/env bash
# EnactToM paper pipeline entry point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PROJECT_ROOT"

RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

COMMAND=""
MODEL="gpt-5.4-mini"
LLM_PROVIDER=""
TASK_FILE=""
TASKS_DIR=""
OUTPUT_DIR=""
NUM_TASKS=1
AGENTS_MIN=2
AGENTS_MAX=4
TASK_GEN_AGENT="mini"
QUERY=""
CATEGORY=""
GENERATION_DIFFICULTY="standard"
TARGET_MODEL="gpt-5.4-mini"
TARGET_PASS_RATE=""
SEED_PASS_RATIO=""
SEED_FAIL_RATIO=""
SEED_TASKS_DIR=""
SAMPLED_TASKS_DIR=""
SUBTASKS_MIN=3
SUBTASKS_MAX=20
ITERATIONS_PER_TASK=200
K_LEVEL=""
NO_ICL=false
THRESHOLD=0.7
JUDGE_DIFFICULTY=""
TEST_MODEL=""
MAX_SIM_STEPS=200000
MAX_LLM_CALLS=""
MAX_WORKERS=""
WORKERS_PER_GPU=""
NUM_GPUS=""
NUM_TIMES=3
BULK_GENERATE=false
RUN_UNTIL=""
DRY_RUN=false
GENERATION_PER_GPU=3
K_DISTRIBUTION=""
NO_VIDEO=true
NO_CALIBRATION=true
OBSERVATION_MODE="text"
BENCHMARK_RUN_MODE="standard"
SELECTOR_MIN_FRAMES=1
SELECTOR_MAX_FRAMES=5
SELECTOR_MAX_CANDIDATES=12
SCENE_DATA_FILE=""
REPORT_FILE=""
RETRY_VERIFICATION=""

usage() {
    cat <<'EOF'
EnactToM paper pipeline

Usage:
  ./enacttom/run.sh <command> [options]

Core commands:
  generate       Generate paper benchmark tasks with the external task authoring agent
  validate-task  Validate task JSON structure without Habitat
  verify-pddl    Check functional PDDL solvability and ToM depth
  verify         Replay a task's golden trajectory in Habitat
  judge          Judge whether a task requires Theory-of-Mind reasoning
  test-task      Run the task acceptance gate used during generation
  new-scene      Create a working task from a sampled Habitat scene
  submit-task    Submit a verified working task into the benchmark task directory
  benchmark      Run standard/baseline benchmark evaluations

Common options:
  --task FILE
  --tasks-dir DIR
  --output-dir DIR
  --model MODEL
  --agents N
  --agents-min N
  --agents-max N
  --category cooperative|mixed
  --difficulty standard|hard
  --bulk
  --per-gpu N
  --run-until N
  --num-tasks N
  --num-times N
  --max-workers N
  --benchmark-run-mode standard|baseline
  --observation-mode text|vision
  --help

Bulk generation:
  ./enacttom/run.sh generate --bulk --num-tasks 24 --per-gpu 3
  ./enacttom/run.sh generate --bulk --run-until 100 --difficulty hard

Examples:
  ./enacttom/run.sh generate --num-tasks 3 --difficulty standard
  ./enacttom/run.sh judge --task data/enacttom/tasks/example.json
  ./enacttom/run.sh benchmark --tasks-dir data/enacttom/tasks --model gpt-5.4-mini --num-times 3
EOF
}

die() {
    echo -e "${RED}Error:${NC} $1" >&2
    echo "Hint: ./enacttom/run.sh --help" >&2
    exit 1
}

python_bin() {
    if [[ -n "${ENACTTOM_PYTHON:-}" ]]; then
        echo "$ENACTTOM_PYTHON"
        return
    fi
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        echo "$CONDA_PREFIX/bin/python"
        return
    fi
    command -v python
}

require_value() {
    local flag=$1
    local value=${2:-}
    if [[ -z "$value" || "$value" == --* ]]; then
        die "$flag requires a value"
    fi
}

normalize_model_alias() {
    case "$1" in
        deepseek) echo "deepseek-v3.2" ;;
        *) echo "$1" ;;
    esac
}

expand_model_name() {
    local model
    model="$(normalize_model_alias "$1")"
    case "$model" in
        kimi-k2.5) echo "accounts/fireworks/models/kimi-k2p5" ;;
        deepseek-v3.2) echo "accounts/fireworks/models/deepseek-v3p2" ;;
        *) echo "$model" ;;
    esac
}

has_anthropic_api_key() {
    [[ -n "${ANTHROPIC_API_KEY:-}" ]] && return 0
    [[ -f ".env" ]] && grep -qE '^[[:space:]]*ANTHROPIC_API_KEY[[:space:]]*=' .env
}

load_anthropic_api_key() {
    if [[ -z "${ANTHROPIC_API_KEY:-}" && -f ".env" ]]; then
        local line value
        line=$(grep -E '^[[:space:]]*ANTHROPIC_API_KEY[[:space:]]*=' .env | tail -n 1 || true)
        value=$(printf '%s' "$line" | sed -E "s/^[^=]+=//; s/^[[:space:]]+//; s/[[:space:]]+$//; s/^['\\\"]//; s/['\\\"]$//")
        [[ -n "$value" ]] && export ANTHROPIC_API_KEY="$value"
    fi
}

detect_llm_provider() {
    local model
    model="$(normalize_model_alias "$1")"
    case "$model" in
        gpt-*|o3|kimi-k2.5|deepseek-v3.2|gemini-*|accounts/fireworks/models/*)
            echo "openai_chat" ;;
        sonnet*|haiku*|opus*)
            if has_anthropic_api_key; then
                echo "anthropic_claude"
            else
                echo "bedrock_claude"
            fi
            ;;
        *)
            echo "" ;;
    esac
}

get_agent_config() {
    local num_agents=$1

    case "$num_agents" in
        2|3|4) echo "examples/enacttom_${num_agents}_robots" ;;
        *) die "--agents must be 2-4 for the EnactToM paper configs" ;;
    esac
}

task_agents() {
    if [[ -n "$TASK_FILE" ]]; then
        local py
        py="$(python_bin)"
        "$py" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("num_agents", 2))' "$TASK_FILE" 2>/dev/null || echo 2
    else
        echo "$AGENTS_MAX"
    fi
}

run_generate() {
    if [[ "$BULK_GENERATE" == true ]]; then
        run_bulk_generate
        return
    fi

    [[ -z "$LLM_PROVIDER" ]] && LLM_PROVIDER="$(detect_llm_provider "$MODEL")"
    [[ -z "$LLM_PROVIDER" ]] && LLM_PROVIDER="external_cli"

    local model_expanded config output_dir py
    model_expanded="$(expand_model_name "$MODEL")"
    config="$(get_agent_config "$AGENTS_MAX")"
    output_dir="${OUTPUT_DIR:-data/enacttom/tasks}"
    py="$(python_bin)"

    local args=()
    [[ -n "$QUERY" ]] && args+=(--query "$QUERY")
    [[ -n "$TASK_GEN_AGENT" ]] && args+=(--task-gen-agent "$TASK_GEN_AGENT")
    [[ -n "$RETRY_VERIFICATION" ]] && args+=(--retry-verification "$RETRY_VERIFICATION")
    [[ -n "$CATEGORY" ]] && args+=(--category "$CATEGORY")
    [[ -n "$TARGET_MODEL" ]] && args+=(--target-model "$(normalize_model_alias "$TARGET_MODEL")")
    [[ -n "$GENERATION_DIFFICULTY" ]] && args+=(--difficulty "$GENERATION_DIFFICULTY")
    [[ -n "$TARGET_PASS_RATE" ]] && args+=(--target-pass-rate "$TARGET_PASS_RATE")
    [[ -n "$SEED_PASS_RATIO" ]] && args+=(--seed-pass-ratio "$SEED_PASS_RATIO")
    [[ -n "$SEED_FAIL_RATIO" ]] && args+=(--seed-fail-ratio "$SEED_FAIL_RATIO")
    [[ -n "$SEED_TASKS_DIR" ]] && args+=(--seed-tasks-dir "$SEED_TASKS_DIR")
    [[ -n "$SAMPLED_TASKS_DIR" ]] && args+=(--sampled-tasks-dir "$SAMPLED_TASKS_DIR")
    [[ "$NO_ICL" == true ]] && args+=(--no-icl)
    [[ -n "$THRESHOLD" ]] && args+=(--judge-threshold "$THRESHOLD")
    [[ -n "$JUDGE_DIFFICULTY" ]] && args+=(--judge-difficulty "$JUDGE_DIFFICULTY")
    [[ -n "$TEST_MODEL" ]] && args+=(--test-model "$(normalize_model_alias "$TEST_MODEL")")
    [[ -n "$K_LEVEL" ]] && args+=(--k-level $K_LEVEL)
    "$py" enacttom/task_gen/runner.py \
        "${args[@]}" \
        --config-name "$config" \
        +num_tasks="$NUM_TASKS" \
        +agents_min="$AGENTS_MIN" \
        +agents_max="$AGENTS_MAX" \
        +model="$model_expanded" \
        +llm_provider="$LLM_PROVIDER" \
        +subtasks_min="$SUBTASKS_MIN" \
        +subtasks_max="$SUBTASKS_MAX" \
        +iterations_per_task="$ITERATIONS_PER_TASK" \
        +output_dir="$output_dir" \
        "hydra.run.dir=./outputs/enacttom/\${now:%Y-%m-%d_%H-%M-%S}-generate"
}

allocate_bulk_output_dir() {
    local run_date day_dir max_generation existing_dir generation_name generation_num
    run_date="$(date +%Y%m%d)"
    day_dir="data/enacttom/tasks/bulk_generation_${run_date}"
    max_generation=0
    if [[ -d "$day_dir" ]]; then
        shopt -s nullglob
        for existing_dir in "$day_dir"/generation_*; do
            [[ -d "$existing_dir" ]] || continue
            generation_name="$(basename "$existing_dir")"
            generation_num="${generation_name#generation_}"
            if [[ "$generation_num" =~ ^[0-9]+$ && "$generation_num" -gt "$max_generation" ]]; then
                max_generation="$generation_num"
            fi
        done
        shopt -u nullglob
    fi
    OUTPUT_DIR="${day_dir}/generation_$((max_generation + 1))"
}

detect_gpu_count() {
    if [[ -n "$NUM_GPUS" ]]; then
        echo "$NUM_GPUS"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local detected
        detected="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$detected" =~ ^[0-9]+$ && "$detected" -gt 0 ]]; then
            echo "$detected"
            return
        fi
    fi
    echo 1
}

count_task_files() {
    local dir=$1
    if [[ ! -d "$dir" ]]; then
        echo 0
        return
    fi
    find "$dir" -maxdepth 1 -name '*.json' -type f | wc -l | tr -d ' '
}

build_bulk_generate_args() {
    BULK_CHILD_ARGS=(generate --num-tasks 1 --model "$MODEL" --task-gen-agent "$TASK_GEN_AGENT")
    BULK_CHILD_ARGS+=(--difficulty "$GENERATION_DIFFICULTY" --output-dir "$OUTPUT_DIR")
    BULK_CHILD_ARGS+=(--agents-min "$AGENTS_MIN" --agents-max "$AGENTS_MAX")
    BULK_CHILD_ARGS+=(--subtasks-min "$SUBTASKS_MIN" --subtasks-max "$SUBTASKS_MAX")
    BULK_CHILD_ARGS+=(--iterations-per-task "$ITERATIONS_PER_TASK")
    [[ -n "$LLM_PROVIDER" ]] && BULK_CHILD_ARGS+=(--llm "$LLM_PROVIDER")
    [[ -n "$QUERY" ]] && BULK_CHILD_ARGS+=(--query "$QUERY")
    [[ -n "$RETRY_VERIFICATION" ]] && BULK_CHILD_ARGS+=(--retry-verification "$RETRY_VERIFICATION")
    [[ -n "$TARGET_MODEL" ]] && BULK_CHILD_ARGS+=(--target-model "$TARGET_MODEL")
    [[ -n "$TARGET_PASS_RATE" ]] && BULK_CHILD_ARGS+=(--target-pass-rate "$TARGET_PASS_RATE")
    [[ -n "$SEED_PASS_RATIO" ]] && BULK_CHILD_ARGS+=(--seed-pass-ratio "$SEED_PASS_RATIO")
    [[ -n "$SEED_FAIL_RATIO" ]] && BULK_CHILD_ARGS+=(--seed-fail-ratio "$SEED_FAIL_RATIO")
    [[ -n "$SEED_TASKS_DIR" ]] && BULK_CHILD_ARGS+=(--seed-tasks-dir "$SEED_TASKS_DIR")
    [[ -n "$SAMPLED_TASKS_DIR" ]] && BULK_CHILD_ARGS+=(--sampled-tasks-dir "$SAMPLED_TASKS_DIR")
    [[ "$NO_ICL" == true ]] && BULK_CHILD_ARGS+=(--no-icl)
    [[ -n "$THRESHOLD" ]] && BULK_CHILD_ARGS+=(--threshold "$THRESHOLD")
    [[ -n "$JUDGE_DIFFICULTY" ]] && BULK_CHILD_ARGS+=(--judge-difficulty "$JUDGE_DIFFICULTY")
    [[ -n "$TEST_MODEL" ]] && BULK_CHILD_ARGS+=(--test-model "$TEST_MODEL")
    return 0
}

parse_k_distribution() {
    BULK_K_LEVELS=()
    [[ -z "$K_DISTRIBUTION" ]] && return
    local pair k_val k_count
    IFS=',' read -ra pairs <<< "$K_DISTRIBUTION"
    for pair in "${pairs[@]}"; do
        k_val="${pair%%:*}"
        k_count="${pair##*:}"
        [[ "$k_val" =~ ^[123]$ && "$k_count" =~ ^[0-9]+$ ]] || die "--k-distribution entries must look like 1:2,2:3"
        for _ in $(seq 1 "$k_count"); do
            BULK_K_LEVELS+=("$k_val")
        done
    done
    [[ "${#BULK_K_LEVELS[@]}" -eq "$GENERATION_PER_GPU" ]] || die "--k-distribution slot counts must sum to --per-gpu"
    return 0
}

run_bulk_worker() {
    local worker_index=$1
    local gpu=$2
    local slot=$3
    local category=$4
    local k_level=$5
    local run_id=$6
    local run_dir=$7
    local total_workers=$8
    local requested_tasks=$9

    local worker_id stdout_log worker_dir
    worker_id="worker-$(printf '%04d' "$worker_index")"
    worker_dir="$run_dir/workers/$worker_id"
    stdout_log="$worker_dir/stdout.log"
    mkdir -p "$worker_dir"

    local cmd=("./enacttom/run.sh" "${BULK_CHILD_ARGS[@]}" --category "$category")
    [[ -n "$k_level" ]] && cmd+=(--k-level $k_level)

    if [[ "$DRY_RUN" == true ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%s ENACTTOM_GENERATION_WORKER_ID=%s ' "$gpu" "$worker_id"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return
    fi

    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export ENACTTOM_GENERATION_MODE="bulk"
        export ENACTTOM_GENERATION_RUN_ID="$run_id"
        export ENACTTOM_GENERATION_RUN_DIR="$run_dir"
        export ENACTTOM_GENERATION_WORKER_ID="$worker_id"
        export ENACTTOM_GENERATION_WORKER_DIR="$worker_dir"
        export ENACTTOM_GENERATION_GPU="$gpu"
        export ENACTTOM_GENERATION_SLOT="$slot"
        export ENACTTOM_GENERATION_TOTAL_WORKERS="$total_workers"
        export ENACTTOM_GENERATION_REQUESTED_TASKS="$requested_tasks"
        export ENACTTOM_GENERATION_STDOUT_LOG="$stdout_log"
        "${cmd[@]}"
    ) >"$stdout_log" 2>&1 &
    BULK_PIDS+=("$!")
}

wait_for_bulk_batch() {
    local failed=0 pid
    for pid in "${BULK_PIDS[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    BULK_PIDS=()
    return "$failed"
}

run_bulk_generate() {
    [[ -z "$OUTPUT_DIR" ]] && allocate_bulk_output_dir
    mkdir -p "$OUTPUT_DIR"

    local gpu_count concurrency requested target before_count run_id run_dir categories
    gpu_count="$(detect_gpu_count)"
    [[ "$gpu_count" =~ ^[0-9]+$ && "$gpu_count" -gt 0 ]] || die "bulk generation needs at least one GPU; use --num-gpus to override detection"

    if [[ -n "$MAX_WORKERS" ]]; then
        concurrency="$MAX_WORKERS"
    else
        concurrency=$((gpu_count * GENERATION_PER_GPU))
    fi
    [[ "$concurrency" -gt 0 ]] || die "bulk generation worker count must be positive"

    requested="$NUM_TASKS"
    [[ -n "$RUN_UNTIL" ]] && requested="$RUN_UNTIL"
    [[ "$requested" =~ ^[0-9]+$ && "$requested" -gt 0 ]] || die "--num-tasks/--run-until must be a positive integer"

    before_count="$(count_task_files "$OUTPUT_DIR")"
    target=$((before_count + requested))
    run_id="bulk-$(date +%Y%m%d-%H%M%S)"
    run_dir="outputs/generations/$run_id"
    mkdir -p "$run_dir/workers"

    if [[ -n "$CATEGORY" ]]; then
        categories=("$CATEGORY")
    else
        categories=("cooperative" "mixed")
    fi

    parse_k_distribution
    build_bulk_generate_args

    printf '%b\n' "${BOLD}Bulk EnactToM generation${NC}"
    echo "Output: $OUTPUT_DIR"
    echo "Run dir: $run_dir"
    echo "GPUs: $gpu_count"
    echo "Concurrency: $concurrency"
    echo "Target new tasks: $requested"

    local launched=0 current_count batch_size i gpu slot category k_level
    while true; do
        current_count="$(count_task_files "$OUTPUT_DIR")"
        [[ "$current_count" -ge "$target" ]] && break
        batch_size=$((target - current_count))
        [[ "$batch_size" -gt "$concurrency" ]] && batch_size="$concurrency"

        BULK_PIDS=()
        for ((i=0; i<batch_size; i++)); do
            gpu=$((i % gpu_count))
            slot=$((i / gpu_count))
            category="${categories[$(((launched + i) % ${#categories[@]}))]}"
            k_level=""
            if [[ "${#BULK_K_LEVELS[@]}" -gt 0 ]]; then
                k_level="${BULK_K_LEVELS[$((slot % ${#BULK_K_LEVELS[@]}))]}"
            elif [[ -n "$K_LEVEL" ]]; then
                k_level="$K_LEVEL"
            fi
            run_bulk_worker "$((launched + i + 1))" "$gpu" "$slot" "$category" "$k_level" "$run_id" "$run_dir" "$concurrency" "$requested"
        done
        launched=$((launched + batch_size))

        [[ "$DRY_RUN" == true ]] && break
        wait_for_bulk_batch || die "one or more bulk generation workers failed; inspect $run_dir/workers/*/stdout.log"
    done

    if [[ "$DRY_RUN" != true ]]; then
        echo "Bulk generation complete: $(count_task_files "$OUTPUT_DIR") task files in $OUTPUT_DIR"
    fi
}

run_benchmark() {
    local model_short model_expanded output_base category_override save_video_override py
    model_short="$(normalize_model_alias "$MODEL")"
    [[ -z "$LLM_PROVIDER" ]] && LLM_PROVIDER="$(detect_llm_provider "$MODEL")"
    [[ -z "$LLM_PROVIDER" ]] && die "unknown model '$MODEL'"
    model_expanded="$(expand_model_name "$MODEL")"
    output_base="${OUTPUT_DIR:-./outputs/enacttom/$(date +%Y-%m-%d_%H-%M-%S)-benchmark}"
    py="$(python_bin)"
    save_video_override=""
    [[ "$NO_VIDEO" == true ]] && save_video_override="++evaluation.save_video=false"
    category_override=""
    [[ -n "$CATEGORY" ]] && category_override="+task_category_filter=$CATEGORY"

    if [[ "$NUM_TIMES" -gt 1 ]]; then
        local repeat_cmd=(
            "$py" -m enacttom.scripts.benchmark_repeat
            --model "$model_short"
            --output-dir "$output_base"
            --num-times "$NUM_TIMES"
            --max-sim-steps "$MAX_SIM_STEPS"
            --benchmark-run-mode "$BENCHMARK_RUN_MODE"
            --observation-mode "$OBSERVATION_MODE"
            --selector-min-frames "$SELECTOR_MIN_FRAMES"
            --selector-max-frames "$SELECTOR_MAX_FRAMES"
            --selector-max-candidates "$SELECTOR_MAX_CANDIDATES"
        )
        if [[ -n "$TASK_FILE" ]]; then
            repeat_cmd+=(--task "$TASK_FILE")
        else
            repeat_cmd+=(--tasks-dir "${TASKS_DIR:-data/enacttom/tasks}")
        fi
        [[ -n "$MAX_LLM_CALLS" ]] && repeat_cmd+=(--max-llm-calls "$MAX_LLM_CALLS")
        [[ -n "$WORKERS_PER_GPU" ]] && repeat_cmd+=(--workers-per-gpu "$WORKERS_PER_GPU")
        [[ -n "$MAX_WORKERS" ]] && repeat_cmd+=(--max-workers "$MAX_WORKERS")
        [[ -n "$NUM_GPUS" ]] && repeat_cmd+=(--num-gpus "$NUM_GPUS")
        [[ -n "$CATEGORY" ]] && repeat_cmd+=(--category "$CATEGORY")
        [[ "$NO_VIDEO" != true ]] && repeat_cmd+=(--video)
        [[ "$NO_CALIBRATION" == true ]] && repeat_cmd+=(--no-calibration) || repeat_cmd+=(--calibration)
        "${repeat_cmd[@]}"
        return
    fi

    if [[ -n "$TASK_FILE" ]]; then
        [[ ! -f "$TASK_FILE" ]] && die "task file not found: $TASK_FILE"
        local num_agents config max_turns_override replanning_overrides
        num_agents="$(task_agents)"
        config="$(get_agent_config "$num_agents")"
        max_turns_override=""
        replanning_overrides=""
        if [[ -n "$MAX_LLM_CALLS" ]]; then
            max_turns_override="+max_turns=$MAX_LLM_CALLS"
            for ((i=0; i<num_agents; i++)); do
                replanning_overrides="$replanning_overrides ++evaluation.agents.agent_${i}.planner.plan_config.replanning_threshold=$MAX_LLM_CALLS"
            done
        fi
        [[ "$LLM_PROVIDER" == "anthropic_claude" ]] && load_anthropic_api_key
        "$py" enacttom/examples/run_habitat_benchmark.py \
            --config-name "$config" \
            habitat.environment.max_episode_steps="$MAX_SIM_STEPS" \
            $max_turns_override \
            $replanning_overrides \
            $save_video_override \
            ++benchmark_observation_mode="$OBSERVATION_MODE" \
            ++benchmark_run_mode="$BENCHMARK_RUN_MODE" \
            $category_override \
            +task="$TASK_FILE" \
            +model="$model_expanded" \
            +llm_provider="$LLM_PROVIDER" \
            "hydra.run.dir=$output_base"
        return
    fi

    local task_dir
    task_dir="${TASKS_DIR:-data/enacttom/tasks}"
    [[ ! -d "$task_dir" ]] && die "task directory not found: $task_dir"

    if [[ -n "$WORKERS_PER_GPU" || -n "$MAX_WORKERS" ]]; then
        local parallel_cmd=(
            "$py" -m enacttom.scripts.run_benchmark_parallel
            --tasks-dir "$task_dir"
            --model "$model_short"
            --output-dir "$output_base"
            --benchmark-run-mode "$BENCHMARK_RUN_MODE"
            --observation-mode "$OBSERVATION_MODE"
            --selector-min-frames "$SELECTOR_MIN_FRAMES"
            --selector-max-frames "$SELECTOR_MAX_FRAMES"
            --selector-max-candidates "$SELECTOR_MAX_CANDIDATES"
        )
        [[ -n "$WORKERS_PER_GPU" ]] && parallel_cmd+=(--workers-per-gpu "$WORKERS_PER_GPU")
        [[ -n "$MAX_WORKERS" ]] && parallel_cmd+=(--max-workers "$MAX_WORKERS")
        [[ "$NO_VIDEO" != true ]] && parallel_cmd+=(--video)
        [[ -n "$CATEGORY" ]] && parallel_cmd+=(--category "$CATEGORY")
        [[ "$NO_CALIBRATION" == true ]] && parallel_cmd+=(--no-calibration) || parallel_cmd+=(--calibration)
        "${parallel_cmd[@]}"
        "$py" -m enacttom.scripts.print_benchmark_summary --output-dir "$output_base" --model "$model_short" --parallel
        return
    fi

    local counts
    counts=$("$py" -c "
import json
from pathlib import Path
category = '$CATEGORY' or None
counts = set()
for path in Path('$task_dir').glob('*.json'):
    data = json.load(open(path))
    if category and data.get('category') != category:
        continue
    counts.add(data.get('num_agents', 2))
print(' '.join(map(str, sorted(counts))))
")
    [[ -z "$counts" ]] && die "no benchmarkable tasks found in $task_dir"

    for num_agents in $counts; do
        local config
        config="$(get_agent_config "$num_agents")"
        [[ "$LLM_PROVIDER" == "anthropic_claude" ]] && load_anthropic_api_key
        "$py" enacttom/examples/run_habitat_benchmark.py \
            --config-name "$config" \
            habitat.environment.max_episode_steps="$MAX_SIM_STEPS" \
            $save_video_override \
            ++benchmark_observation_mode="$OBSERVATION_MODE" \
            ++benchmark_run_mode="$BENCHMARK_RUN_MODE" \
            $category_override \
            +num_agents_filter="$num_agents" \
            +task_dir="$task_dir" \
            +model="$model_expanded" \
            +llm_provider="$LLM_PROVIDER" \
            "hydra.run.dir=${output_base}-${num_agents}agents"
    done
    "$py" -m enacttom.scripts.print_benchmark_summary --output-dir "$output_base" --model "$model_short"
}

run_judge() {
    [[ -z "$TASK_FILE" ]] && die "--task is required"
    local args=("$TASK_FILE" --threshold "$THRESHOLD")
    [[ -n "$JUDGE_DIFFICULTY" ]] && args+=(--difficulty "$JUDGE_DIFFICULTY")
    "$(python_bin)" -m enacttom.cli.judge_task "${args[@]}"
}

run_verify() {
    [[ -z "$TASK_FILE" ]] && die "--task is required"
    [[ ! -f "$TASK_FILE" ]] && die "task file not found: $TASK_FILE"
    local num_agents config result_file workdir log_file py
    num_agents="$(task_agents)"
    config="$(get_agent_config "$num_agents")"
    result_file="${REPORT_FILE:-/tmp/enacttom_verify_$(date +%Y%m%d_%H%M%S)_$$.json}"
    workdir="${OUTPUT_DIR:-/tmp}"
    log_file="${result_file%.json}.log"
    py="$(python_bin)"
    mkdir -p "$(dirname "$result_file")" "$workdir"

    set +e
    "$py" -m enacttom.cli.verify_trajectory \
        "$TASK_FILE" \
        --working-dir "$workdir" \
        --config-name "$config" \
        >"$result_file" 2>"$log_file"
    local exit_code=$?
    set -e

    cat "$log_file"
    "$py" - <<PY
import json, re
from pathlib import Path
p = Path("$result_file")
if not p.exists():
    data = {"valid": False, "error": "verification failed before producing JSON", "exit_code": $exit_code}
else:
    raw = p.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
        for candidate in reversed(re.findall(r"\{.*\}", raw, flags=re.DOTALL)):
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                pass
        data = data or {"valid": False, "error": "verification output did not contain parseable JSON"}
print(json.dumps(data, indent=2))
valid = bool(data.get("valid", data.get("success", False)))
raise SystemExit(0 if valid else 1)
PY
}

run_verify_pddl() {
    [[ -z "$TASK_FILE" ]] && die "--task is required"
    local args=("$TASK_FILE")
    [[ -n "$OUTPUT_DIR" ]] && args+=(--working-dir "$OUTPUT_DIR")
    "$(python_bin)" -m enacttom.cli.verify_pddl "${args[@]}"
}

run_validate_task() {
    [[ -z "$TASK_FILE" ]] && die "--task is required"
    local args=("$TASK_FILE")
    [[ -n "$SCENE_DATA_FILE" ]] && args+=(--scene-file "$SCENE_DATA_FILE")
    "$(python_bin)" -m enacttom.cli.validate_task "${args[@]}"
}

run_test_task() {
    [[ -z "$TASK_FILE" ]] && die "--task is required"
    local args=("$TASK_FILE")
    [[ -n "$OUTPUT_DIR" ]] && args+=(--working-dir "$OUTPUT_DIR" --trajectory-dir "$OUTPUT_DIR/trajectories")
    [[ -n "$TEST_MODEL" ]] && args+=(--test-model "$TEST_MODEL")
    [[ -n "$BENCHMARK_RUN_MODE" ]] && args+=(--run-mode "$BENCHMARK_RUN_MODE")
    "$(python_bin)" -m enacttom.cli.test_task "${args[@]}"
}

run_new_scene() {
    "$(python_bin)" -m enacttom.cli.new_scene "${AGENTS_MAX:-2}" --working-dir "${OUTPUT_DIR:-/tmp/enacttom_scene}"
}

run_submit_task() {
    [[ -z "$TASK_FILE" ]] && die "--task is required"
    "$(python_bin)" -m enacttom.cli.submit_task "$TASK_FILE" --output-dir "${OUTPUT_DIR:-data/enacttom/tasks}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        generate|validate-task|verify-pddl|verify|judge|test-task|new-scene|submit-task|benchmark)
            COMMAND="$1"; shift ;;
        -h|--help)
            usage; exit 0 ;;
        --task)
            require_value "$1" "${2:-}"; TASK_FILE="$2"; shift 2 ;;
        --tasks-dir|--task-dir)
            require_value "$1" "${2:-}"; TASKS_DIR="$2"; shift 2 ;;
        --output-dir)
            require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --model)
            require_value "$1" "${2:-}"; MODEL="$2"; shift 2 ;;
        --llm)
            require_value "$1" "${2:-}"; LLM_PROVIDER="$2"; shift 2 ;;
        --num-tasks|--tasks)
            require_value "$1" "${2:-}"; NUM_TASKS="$2"; shift 2 ;;
        --bulk)
            BULK_GENERATE=true; shift ;;
        --per-gpu)
            require_value "$1" "${2:-}"; GENERATION_PER_GPU="$2"; shift 2 ;;
        --run-until)
            require_value "$1" "${2:-}"; RUN_UNTIL="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --agents)
            require_value "$1" "${2:-}"; AGENTS_MIN="$2"; AGENTS_MAX="$2"; shift 2 ;;
        --agents-min)
            require_value "$1" "${2:-}"; AGENTS_MIN="$2"; shift 2 ;;
        --agents-max|--num-agents)
            require_value "$1" "${2:-}"; AGENTS_MAX="$2"; shift 2 ;;
        --task-gen-agent)
            require_value "$1" "${2:-}"; TASK_GEN_AGENT="$2"; shift 2 ;;
        --query)
            require_value "$1" "${2:-}"; QUERY="$2"; shift 2 ;;
        --retry-verification)
            require_value "$1" "${2:-}"; RETRY_VERIFICATION="$2"; shift 2 ;;
        --category)
            require_value "$1" "${2:-}"; CATEGORY="$2"; shift 2 ;;
        --difficulty)
            require_value "$1" "${2:-}"; GENERATION_DIFFICULTY="$2"; shift 2 ;;
        --target-model)
            require_value "$1" "${2:-}"; TARGET_MODEL="$2"; shift 2 ;;
        --target-pass-rate)
            require_value "$1" "${2:-}"; TARGET_PASS_RATE="$2"; shift 2 ;;
        --seed-pass-ratio)
            require_value "$1" "${2:-}"; SEED_PASS_RATIO="$2"; shift 2 ;;
        --seed-fail-ratio)
            require_value "$1" "${2:-}"; SEED_FAIL_RATIO="$2"; shift 2 ;;
        --seed-tasks-dir)
            require_value "$1" "${2:-}"; SEED_TASKS_DIR="$2"; shift 2 ;;
        --sampled-tasks-dir)
            require_value "$1" "${2:-}"; SAMPLED_TASKS_DIR="$2"; shift 2 ;;
        --subtasks)
            require_value "$1" "${2:-}"; SUBTASKS_MIN="$2"; SUBTASKS_MAX="$2"; shift 2 ;;
        --subtasks-min)
            require_value "$1" "${2:-}"; SUBTASKS_MIN="$2"; shift 2 ;;
        --subtasks-max)
            require_value "$1" "${2:-}"; SUBTASKS_MAX="$2"; shift 2 ;;
        --iterations-per-task)
            require_value "$1" "${2:-}"; ITERATIONS_PER_TASK="$2"; shift 2 ;;
        --k-level)
            shift
            K_LEVEL=""
            while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do
                K_LEVEL="${K_LEVEL:+$K_LEVEL }$1"
                shift
            done
            [[ -z "$K_LEVEL" ]] && die "--k-level requires at least one integer" ;;
        --k-distribution)
            require_value "$1" "${2:-}"; K_DISTRIBUTION="$2"; shift 2 ;;
        --remove)
            die "--remove is no longer supported; run the full pipeline" ;;
        --no-icl)
            NO_ICL=true; shift ;;
        --threshold)
            require_value "$1" "${2:-}"; THRESHOLD="$2"; shift 2 ;;
        --judge-difficulty)
            require_value "$1" "${2:-}"; JUDGE_DIFFICULTY="$2"; shift 2 ;;
        --test-model)
            require_value "$1" "${2:-}"; TEST_MODEL="$2"; shift 2 ;;
        --max-sim-steps)
            require_value "$1" "${2:-}"; MAX_SIM_STEPS="$2"; shift 2 ;;
        --max-llm-calls)
            require_value "$1" "${2:-}"; MAX_LLM_CALLS="$2"; shift 2 ;;
        --max-workers)
            require_value "$1" "${2:-}"; MAX_WORKERS="$2"; shift 2 ;;
        --workers-per-gpu)
            require_value "$1" "${2:-}"; WORKERS_PER_GPU="$2"; shift 2 ;;
        --num-gpus)
            require_value "$1" "${2:-}"; NUM_GPUS="$2"; shift 2 ;;
        --num-times)
            require_value "$1" "${2:-}"; NUM_TIMES="$2"; shift 2 ;;
        --video)
            NO_VIDEO=false; shift ;;
        --calibration)
            NO_CALIBRATION=false; shift ;;
        --no-calibration)
            NO_CALIBRATION=true; shift ;;
        --observation-mode)
            require_value "$1" "${2:-}"; OBSERVATION_MODE="$2"; shift 2 ;;
        --benchmark-run-mode)
            require_value "$1" "${2:-}"; BENCHMARK_RUN_MODE="$2"; shift 2 ;;
        --selector-min-frames)
            require_value "$1" "${2:-}"; SELECTOR_MIN_FRAMES="$2"; shift 2 ;;
        --selector-max-frames)
            require_value "$1" "${2:-}"; SELECTOR_MAX_FRAMES="$2"; shift 2 ;;
        --selector-max-candidates)
            require_value "$1" "${2:-}"; SELECTOR_MAX_CANDIDATES="$2"; shift 2 ;;
        --scene-data)
            require_value "$1" "${2:-}"; SCENE_DATA_FILE="$2"; shift 2 ;;
        --report-file)
            require_value "$1" "${2:-}"; REPORT_FILE="$2"; shift 2 ;;
        *)
            die "unknown command or option '$1'" ;;
    esac
done

[[ -z "$COMMAND" ]] && { usage; exit 1; }

[[ ! "$AGENTS_MIN" =~ ^[0-9]+$ || "$AGENTS_MIN" -lt 2 || "$AGENTS_MIN" -gt 4 ]] && die "--agents-min must be 2-4"
[[ ! "$AGENTS_MAX" =~ ^[0-9]+$ || "$AGENTS_MAX" -lt 2 || "$AGENTS_MAX" -gt 4 ]] && die "--agents-max must be 2-4"
[[ "$AGENTS_MIN" -gt "$AGENTS_MAX" ]] && die "--agents-min cannot exceed --agents-max"
[[ -n "$CATEGORY" && "$CATEGORY" != "cooperative" && "$CATEGORY" != "mixed" ]] && die "--category must be cooperative or mixed"
[[ "$GENERATION_DIFFICULTY" != "standard" && "$GENERATION_DIFFICULTY" != "hard" ]] && die "--difficulty must be standard or hard"
[[ "$OBSERVATION_MODE" != "text" && "$OBSERVATION_MODE" != "vision" ]] && die "--observation-mode must be text or vision"
[[ "$BENCHMARK_RUN_MODE" != "standard" && "$BENCHMARK_RUN_MODE" != "baseline" ]] && die "--benchmark-run-mode must be standard or baseline"
[[ "$TASK_GEN_AGENT" != "mini" && "$TASK_GEN_AGENT" != "claude" && "$TASK_GEN_AGENT" != "codex" ]] && die "--task-gen-agent must be mini, claude, or codex"
[[ ! "$NUM_TASKS" =~ ^[0-9]+$ || "$NUM_TASKS" -lt 1 ]] && die "--num-tasks must be an integer >= 1"
[[ ! "$GENERATION_PER_GPU" =~ ^[0-9]+$ || "$GENERATION_PER_GPU" -lt 1 ]] && die "--per-gpu must be an integer >= 1"
if [[ -n "$RUN_UNTIL" ]]; then
    [[ "$RUN_UNTIL" =~ ^[0-9]+$ && "$RUN_UNTIL" -ge 1 ]] || die "--run-until must be an integer >= 1"
fi

case "$COMMAND" in
    generate) run_generate ;;
    validate-task) run_validate_task ;;
    verify-pddl) run_verify_pddl ;;
    verify) run_verify ;;
    judge) run_judge ;;
    test-task) run_test_task ;;
    new-scene) run_new_scene ;;
    submit-task) run_submit_task ;;
    benchmark) run_benchmark ;;
esac

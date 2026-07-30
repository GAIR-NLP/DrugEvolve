<h1 align="center">DrugEvolve</h1>

<p align="center">
  <strong>An AI System for Autonomous Algorithm Evolution in Drug Development</strong>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Version 0.1.0" src="https://img.shields.io/badge/version-0.1.0-6C63FF">
  <img alt="Status Alpha" src="https://img.shields.io/badge/status-alpha-F59E0B">
  <img alt="License Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-2EA44F">
</p>

<p align="center">
  <img src="docs/drugevolve.jpg" alt="DrugEvolve framework" width="95%">
</p>

DrugEvolve is a multi-role framework for autonomously designing, implementing, evaluating and refining algorithms for drug development, with broader applicability to algorithmic innovation across scientific domains.


Starting from a user-defined task and training entry point, DrugEvolve
coordinates three specialized domains:

| Domain | Responsibility |
| --- | --- |
| **Researcher** | Samples previous experiments, proposes new algorithmic ideas, checks novelty, and implements candidate code |
| **Engineer** | Runs the candidate, monitors training, debugs failures, and calculates objective and LLM-assisted scores |
| **Analyst** | Compares results with the baseline and related experiments, then distills reusable evolutionary experience |

Experiment history, candidate lineage, sampler state, external cognition, and
best snapshots are stored locally so an evolution run remains inspectable and
resumable.


## Contents

- [What DrugEvolve does](#what-drugevolve-does)
- [Included applications](#included-applications)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Connect your task](#connect-your-task)
- [Customize the agent prompts](#customize-the-agent-prompts)
- [Configure an evolution run](#configure-an-evolution-run)
- [Sampling strategies](#sampling-strategies)
- [Literature knowledge as cognition](#literature-knowledge-as-cognition)
- [Outputs and run state](#outputs-and-run-state)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Development](#development)
- [License](#license)

## What DrugEvolve does

For every evolution round, the framework:

1. samples a parent candidate and reference experiments;
2. retrieves related external cognition when available;
3. asks the Researcher to propose a novel algorithm;
4. asks the Implementer to materialize the proposal as `model.py`;
5. executes the task-specific training script;
6. retries failed candidates through the Debugger;
7. combines objective evaluation with an optional LLM score;
8. asks the Analyst to interpret the result;
9. stores the candidate, score, analysis, lineage, and reusable experience;
10. updates the best snapshot and continues until a stop condition is met.

DrugEvolve keeps two kinds of memory separate:

- **Evolutionary database:** candidate code, scores, failures, analyses,
  parents, visit counts, and lineage.
- **Cognition store:** external knowledge such as literature, scientific
  heuristics, and expert guidance.

The current compatibility pipeline runs experiments locally on one machine.
GPU selection is controlled through `CUDA_DEVICE`, and training subprocesses
are bounded by a configurable timeout.

## Included applications

<p align="center">
  <img src="docs/drugevolve-applications.jpg" alt="DrugEvolve framework" width="95%">
</p>

The [`drugevolve-applications/`](drugevolve-applications/) directory contains the baseline and evolved implementations corresponding to the tasks evaluated in the DrugEvolve study. For each application, the directory provides the code before and after autonomous evolution, organized across four stages of drug development:

<table>
  <thead>
    <tr>
      <th width="28%">Stage</th>
      <th width="72%">Included tasks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Target identification</strong></td>
      <td>Cancer gene module detection; gene-disease association prediction; protein druggable site annotation</td>
    </tr>
    <tr>
      <td><strong>Drug discovery</strong></td>
      <td>Molecular docking; drug-target interaction prediction; de novo molecular generation; target-specific peptide binder design; ADMET properties prediction</td>
    </tr>
    <tr>
      <td><strong>Preclinical study</strong></td>
      <td>Animal toxicity assessment</td>
    </tr>
    <tr>
      <td><strong>Clinical trial</strong></td>
      <td>Drug-drug interaction prediction; drug side-effect prediction</td>
    </tr>
  </tbody>
</table>



## Requirements

- Python 3.10+
- A task-specific training environment
- An OpenAI-compatible API key and endpoint
- The computational resources required to run the corresponding baseline implementation

DrugEvolve itself does not prescribe a particular ML framework. Your connected
task may use PyTorch, PyTorch Geometric, RDKit, Transformers, or another stack.

## Quick start

Running DrugEvolve requires four pieces:

1. a task directory with a training launcher;
2. task-specific output and scoring behavior;
3. a confirmed run specification;
4. API and runtime environment variables.

The following commands show the complete setup sequence. Detailed contracts
for each file are described in the next sections.

### 1. Install DrugEvolve

```bash
python -m pip install -e '.[agents,dev]'
```

### 2. Create a task workspace

```text
tasks/my_task/src/
├── run.sh
├── train.py
└── algorithm/
```

| Path | Purpose |
| --- | --- |
| `run.sh` | Required training-launcher template. DrugEvolve copies it into each candidate directory, rewrites its `MODEL_PATH`, and executes it to train and evaluate the generated candidate. |
| `train.py` | Example task-specific training entry point called by `run.sh`. It should load the dataset, import or execute the candidate at `MODEL_PATH`, run training/evaluation, and write the required result files. The file name is not fixed: it may be replaced by another script or command as long as `run.sh` launches it correctly. |
| `algorithm/` | Writable candidate workspace. DrugEvolve creates one subdirectory per candidate and stores the generated implementation and execution outputs there. |

The Implementer creates
`algorithm/<candidate_name>/model.py` for every new candidate.

### 3. Create a run specification

Save the YAML configuration shown in
[Configure an evolution run](#configure-an-evolution-run) as, for example:

```text
configs/my_task.yaml
```

Initialize and confirm the run:

```bash
RUN_DIR=".drugevolve/runs/my_task_run"

drugevolve init \
  --run-dir "$RUN_DIR" \
  --workspace-root . \
  --spec configs/my_task.yaml

drugevolve preflight \
  --run-dir "$RUN_DIR" \
  --workspace-root .

drugevolve preflight \
  --run-dir "$RUN_DIR" \
  --workspace-root . \
  --confirm
```

Always inspect `"$RUN_DIR/preflight.md"` before confirmation.

### 4. Configure the runtime

Run these commands from the repository root:

```bash
export DRUGEVOLVE_API_KEY="<your-api-key>"
export DRUGEVOLVE_BASE_URL="<your-base-url>"

export DRUGEVOLVE_ROOT="$PWD"
export DRUGEVOLVE_TASK_NAME="my_task"
export DRUGEVOLVE_SOURCE_FILE="$PWD/tasks/my_task/src/run.sh"
export DRUGEVOLVE_CODE_DIR="$PWD/tasks/my_task/src"
export DRUGEVOLVE_LOGS_DIR="$PWD/tasks/my_task/src"

export DRUGEVOLVE_RUN_NAME="my_task_run"
export DRUGEVOLVE_RUN_SPEC="$PWD/.drugevolve/runs/my_task_run/run_spec.yaml"
export DRUGEVOLVE_BASELINE_CONTENT="Describe the baseline and its reference metrics here."

export CUDA_DEVICE=0
```

### 5. Start the evolution pipeline

```bash
bash pipeline/run.sh
```

The number of rounds, patience, sampler, and failure limit are loaded from the
confirmed run specification.

## Connect your task

### Task workspace

DrugEvolve expects a writable task workspace similar to:

```text
tasks/<task_name>/src/
├── run.sh
├── train.py
├── data/
└── algorithm/
```

Only `run.sh` is a framework-level requirement. The remaining layout is owned
by your task.

Set the corresponding environment variables:

```bash
export DRUGEVOLVE_TASK_NAME="<task_name>"
export DRUGEVOLVE_TASKS_DIR="$PWD/tasks"
export DRUGEVOLVE_SOURCE_FILE="$PWD/tasks/<task_name>/src/run.sh"
export DRUGEVOLVE_CODE_DIR="$PWD/tasks/<task_name>/src"
export DRUGEVOLVE_LOGS_DIR="$PWD/tasks/<task_name>/src"
export DRUGEVOLVE_DATASET="<dataset_name>"
```

### Training launcher contract

For each candidate, DrugEvolve:

1. creates `CODE_DIR/algorithm/<candidate_name>/model.py`;
2. asks the Implementer to replace the starter file with candidate code;
3. copies the configured `run.sh` into the candidate directory as
   `launch_bash.sh`;
4. rewrites the quoted `MODEL_PATH` assignment to the candidate `model.py`;
5. executes `launch_bash.sh` from inside the candidate directory.

Your `run.sh` should therefore contain a quoted `MODEL_PATH` assignment that
ends in `model.py`:

```bash
#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${DRUGEVOLVE_CODE_DIR}/algorithm/baseline/model.py"

python "$DRUGEVOLVE_CODE_DIR/train.py" \
  --model_path "$MODEL_PATH" \
  --train_file "$DRUGEVOLVE_CODE_DIR/data/train.csv" \
  --val_file "$DRUGEVOLVE_CODE_DIR/data/val.csv" \
  --test_file "$DRUGEVOLVE_CODE_DIR/data/test.csv" \
  --output_dir "$(pwd)"
```

Because the candidate script is executed with the candidate directory as its
working directory, `--output_dir "$(pwd)"` places outputs where the default
collector can find them.

### Candidate output contract

The default Engineer collects these files from the candidate directory or its
immediate `default/` subdirectory:

| File | Required | Purpose |
| --- | --- | --- |
| `*_test.csv` | Yes | Test metrics used by the default objective scorer |
| `*_metric.csv` | Optional | Training or validation metric history |
| `score_detail.json` | Optional | Structured per-metric details |
| `case_study.txt` | Optional | Qualitative samples for Analyst review |

Example test output:

```csv
step,roc_auc,pr_auc
1,0.781,0.694
```

The built-in objective scorer is provided only as a placeholder example to illustrate how evaluation results can be converted into a scalar fitness score. It should not be treated as a general scoring protocol. 

Before running DrugEvolve on a specific task, users should define an appropriate objective function based on the task-specific metrics, optimization directions, constraints and evaluation criteria by adapting the relevant hooks in `pipeline/utils/experiment.py`:

- `collect_algorithm_results`
- `compute_test_score`
- `evaluate_algorithm`
- `parse_train_log`, when entropy monitoring needs task-specific logs

### Baseline and score composition

Provide the baseline description and reference metrics to the Judger:

```bash
export DRUGEVOLVE_BASELINE_CONTENT="Baseline: GNN with ROC-AUC 0.74 and PR-AUC 0.65."
```

By default, the final selection score uses only the objective score:

```bash
export DRUGEVOLVE_OBJECTIVE_SCORE_WEIGHT=1.0
export DRUGEVOLVE_LLM_SCORE_WEIGHT=0.0
```

To blend an LLM Judger score into candidate selection, set both weights
explicitly. At least one weight must be positive.

## Customize the agent prompts

Built-in prompts are intentionally generic. A task prompt profile lets you
specify biological constraints, expected metrics, allowed modules, code
interfaces, and output requirements without changing the orchestration code.

Create a directory such as:

```text
prompts/my_task/
├── generator.txt
├── generator_duplicate.txt
├── inspector.txt
├── implementer.txt
├── debugger.txt
├── judger.txt
├── analyzer.txt
└── summarizer.txt
```

Then set:

```bash
export DRUGEVOLVE_PROMPT_DIR="$PWD/prompts/my_task"
```

Each file is a UTF-8 `string.Template` document and therefore uses
`$variable` placeholders. Missing profile files fall back to built-in prompts.

| Profile | What to describe | Available variables |
| --- | --- | --- |
| `generator.txt` | Task objective, biological priors, hard constraints, acceptable algorithm families, required JSON output | `$parent`, `$references`, `$context` |
| `generator_duplicate.txt` | How to escape repeated design patterns while retaining task feasibility | `$parent`, `$references`, `$repeated_context`, `$context` |
| `inspector.txt` | Task-specific novelty criteria and what counts as a duplicate | `$motivation`, `$explain`, `$math`, `$context` |
| `implementer.txt` | Framework, CLI arguments, data schema, file boundaries, permitted dependencies, required outputs | `$name`, `$motivation`, `$explain`, `$math` |
| `debugger.txt` | Failure diagnostics, protected behavior, allowed fixes, and retry expectations | `$name`, `$motivation`, `$explain`, `$math`, `$previous_error` |
| `judger.txt` | Metric direction, baseline values, weights, validity checks, and score rubric | `$motivation`, `$explain`, `$math`, `$results`, `$baseline` |
| `analyzer.txt` | How to interpret metrics, reference experiments, trade-offs, and qualitative cases | `$name`, `$motivation`, `$explain`, `$math`, `$results`, `$references`, `$case_study` |
| `summarizer.txt` | Which lessons should be retained for future rounds | `$motivation`, `$explain`, `$math`, `$analysis`, `$cognition` |

Example `generator.txt`:

```text
You are designing an algorithm for an ADMET prediction task.

The candidate must:
- use the provided molecular graph input;
- remain compatible with the existing training CLI;
- optimize ROC-AUC and PR-AUC;
- avoid data leakage across scaffold splits;
- fit on one configured GPU.

Parent algorithm:
$parent

Related experiments and external cognition:
$references

Propose a materially different but trainable method.
Return only JSON with: name, motivation, explain, and math.
```

Only the variables listed above are supplied by the current pipeline.
Task-specific constants should be written directly into the profile.

## Configure an evolution run

DrugEvolve requires a YAML run specification and explicit preflight
confirmation.

```yaml
objective: "Improve ROC-AUC and PR-AUC on the scaffold-split validation set"

evaluation:
  command: "bash tasks/my_task/src/run.sh"
  core_score: score
  direction: maximize
  secondary_metrics:
    - roc_auc
    - pr_auc
  timeout_secs: 7200
  success_criteria:
    - "candidate training exits successfully"
    - "candidate writes a non-empty test CSV"

budget:
  max_rounds: 20
  patience: 5
  max_consecutive_failures: 3

stop_conditions:
  - "maximum rounds reached"
  - "patience exhausted"
  - "consecutive failure limit reached"

mutation_scope:
  writable_paths:
    - tasks/my_task/src
  primary_targets:
    - tasks/my_task/src/algorithm

sampling:
  algorithm: ucb1
  sample_n: 3
  exploration_coefficient: 1.414
  num_islands: 4
  exploration_ratio: 0.2
  exploitation_ratio: 0.3
  feature_dimensions:
    - complexity
    - diversity
  feature_bins: 10

cognition:
  source_mode: manual
  seed_files: []

confirmed: false
```

> [!NOTE]
> The compatibility pipeline executes `DRUGEVOLVE_SOURCE_FILE` through its
> Engineer stage. `evaluation.command` is still required by the shared run-spec
> and preflight schema; use it to document the equivalent task evaluator.

Initialize the run:

```bash
RUN_DIR=".drugevolve/runs/my_task_run"

drugevolve init \
  --run-dir "$RUN_DIR" \
  --workspace-root . \
  --spec configs/my_task.yaml
```

Review the generated files:

```bash
drugevolve preflight \
  --run-dir "$RUN_DIR" \
  --workspace-root .
```

Confirm only after reviewing `preflight.md`:

```bash
drugevolve preflight \
  --run-dir "$RUN_DIR" \
  --workspace-root . \
  --confirm
```

The pipeline loads `max_rounds`, `patience`, failure limits, and sampler
settings from this confirmed specification.

## Sampling strategies

The multi-agent pipeline supports the following parent-sampling modes through
the shared experiment store:

| Strategy | Selection behavior | Recommended use |
| --- | --- | --- |
| `ucb1` | Combines normalized candidate utility with a visit-count exploration bonus | A reliable scalar score is available, but candidates have unequal retrieval histories |
| `island` | Rotates across islands, preserves per-niche elites, and mixes random, weighted, and archive selection | The search space is heterogeneous, multimodal, or diversity-sensitive |

### UCB1

```yaml
sampling:
  algorithm: ucb1
  sample_n: 3
  exploration_coefficient: 1.414
```

Unvisited candidates are prioritized. After candidates have been visited, the
sampler balances normalized score quality against an exploration bonus.

### Island sampling

```yaml
sampling:
  algorithm: island
  sample_n: 3
  num_islands: 4
  exploration_ratio: 0.2
  exploitation_ratio: 0.3
  feature_dimensions: [complexity, diversity]
  feature_bins: 10
```

Built-in feature values include candidate-code length as `complexity` and
code-difference-based `diversity`. Numeric evaluator metrics can also be used
as feature dimensions.

Sampling configuration is persisted with the database. Start a new run when
changing sampling strategies after candidates have already been recorded.

## Literature knowledge as cognition

External literature notes can be stored separately from experiment outcomes:

```bash
RUN_DIR=".drugevolve/runs/my_task_run"

drugevolve cognition-add \
  --run-dir "$RUN_DIR" \
  --source paper \
  --kind method \
  --content "Scaffold-aware splitting reduces leakage in molecular property prediction."
```

Verify retrieval:

```bash
drugevolve cognition-search \
  --run-dir "$RUN_DIR" \
  --query "molecular property scaffold split" \
  --top-k 5
```

During evolution, the pipeline searches cognition using the sampled parent's
motivation, explanation, and mathematical description. Up to three matching
items are injected into the Generator context and later made available to the
Analyst summarizer.

## Configuration reference

### Essential variables

| Variable | Purpose |
| --- | --- |
| `DRUGEVOLVE_API_KEY` | API key for the OpenAI-compatible endpoint |
| `DRUGEVOLVE_BASE_URL` | OpenAI-compatible API base URL |
| `DRUGEVOLVE_ROOT` | Absolute repository root |
| `DRUGEVOLVE_TASK_NAME` | Name of the connected task |
| `DRUGEVOLVE_SOURCE_FILE` | Task `run.sh` template |
| `DRUGEVOLVE_CODE_DIR` | Writable task source and candidate workspace |
| `DRUGEVOLVE_RUN_NAME` | Durable run identifier |
| `DRUGEVOLVE_RUN_SPEC` | Confirmed run-spec path |

### Common optional variables

| Variable | Default | Purpose |
| --- | ---: | --- |
| `CUDA_DEVICE` | `0` | GPU made visible to local candidate training |
| `DRUGEVOLVE_DEFAULT_MODEL` | `gpt-4o-mini` | Default model for every agent role |
| `DRUGEVOLVE_AGENT_MODELS` | empty | JSON map overriding models by role |
| `DRUGEVOLVE_PROMPT_DIR` | empty | Task-specific prompt profile directory |
| `DRUGEVOLVE_DATASET` | `default` | Dataset label used by task utilities |
| `DRUGEVOLVE_TRAINING_TIMEOUT` | `7200` | Candidate training timeout in seconds |
| `DRUGEVOLVE_OBJECTIVE_SCORE_WEIGHT` | `1.0` | Weight of task-derived score |
| `DRUGEVOLVE_LLM_SCORE_WEIGHT` | `0.0` | Weight of LLM Judger score |
| `DRUGEVOLVE_LOG_CONTENT` | `0` | Set to `1` to retain prompt/output bodies in trusted environments |

All defaults and sanity checks are defined in `pipeline/env.sh`.

## Security

- Review the generated preflight before confirming mutation or evaluation.
- Agent writes are restricted to `CODE_DIR/algorithm`.
- Candidate training scripts must stay inside `CODE_DIR`.
- Keep API keys in environment variables and never commit `.env` files.
- Prompt and output bodies are redacted from logs by default.
- Set `DRUGEVOLVE_LOG_CONTENT=1` only in a trusted environment.
- Review all user-authored shell commands before execution.

See [SECURITY.md](SECURITY.md) for the full security model.

## Development

```bash
python -m pytest
ruff check drugevolve tests
python -m build
```

Additional project documentation:

- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## License

DrugEvolve is released under the [Apache License 2.0](LICENSE).

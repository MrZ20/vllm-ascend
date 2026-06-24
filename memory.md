# Temporary Memory

This file preserves the current task context for later continuation if chat
context is compacted.

## User Goal

Finish the nightly refactor work on branch `ci_nightly`, using the old
`nightly_refactor` branch only as a reference for intent. The old branch is
stale, so workflow content and test additions/deletions should be based on the
current `ci_nightly` branch.

## Global Rules From User

- Work step by step.
- Before each step, use the `grill-me` skill to ask one focused question.
- After each step, stop and wait for user review before continuing.
- Comment-triggered functionality can remain incompatible for now, but keep the
  reason visible. It will be unified later.
- Local tests should run in the repo virtual environment (`.venv`) when needed.
- Do not blindly copy stale `nightly_refactor` changes.

## Confirmed Scope

The scheduled e2e refactor should converge scheduled tests into:

```text
tests/e2e/schedule/{accuracy,model,ops,scripts}
```

Do not copy all stale `tests/ut` or `tests/e2e/pull_request` changes from
`nightly_refactor`. Only migrate or adjust those areas if they are directly
required by the scheduled e2e refactor.

## Original Step Plan

1. Observe `nightly_refactor` test directory structure changes to understand
   intent, without modifying `ci_nightly`.
2. Rewrite `.github/workflows/scripts/parse_schedule_config.py` according to
   `.github/workflows/scripts/parse_schedule_config_v2.md`, using
   `.github/workflows/scripts/parse_schedule_config.py` as reference.
3. Complete `.github/workflows/_e2e_periodic_ops.yaml`; it was extracted from
   `.github/workflows/schedule_weekly_test_a2.yaml`. The old branch workflow is
   wrong, so current `ci_nightly` should drive the workflow shape.
4. Unify lm_eval accuracy testing into the scheduled test logic:
   remove `tests/e2e/models/configs/accuracy_groups_a2.json`, move grouping
   into `.github/workflows/scripts/schedule_config.yaml`, and route runners
   from paths.
5. Complete `tests/` directory migration.
6. Leave comment-trigger compatibility unfinished for now, but document why.

## Completed Step 1

Observed `nightly_refactor` without changing files.

Findings:

- Old intent is to move scheduled e2e assets into `tests/e2e/schedule/`.
- `tests/e2e/schedule/accuracy`: accuracy YAML configs grouped by
  `one_card/two_card/four_card` and `a2/a3`.
- `tests/e2e/schedule/model`: nightly/weekly single-node and multi-node model
  configs grouped by model family and resource shape.
- `tests/e2e/schedule/ops`: nightly ops tests grouped by resource shape.
- `tests/e2e/schedule/scripts`: execution scripts grouped into
  `accuracy`, `single_node`, and `multi_node`.
- `accuracy_groups_a2.json` and `accuracy.txt` should be removed in favor of
  unified `schedule_config.yaml`.

User confirmed this scope.

## Step 2 Status

The file `.github/workflows/scripts/parse_schedule_config.py` was created and
then refactored after user feedback.

Current user feedback for this file:

- Do not over-factor into too many tiny functions.
- Prefer a readable main flow with comments where appropriate.
- Allow some duplication for readability and future extensibility by test type
  (`single_node`, `multi_node`, etc.).
- Keep only one meaningful parameter/field set. Do not support old aliases such
  as `section.name`, `--schedule-name`, or `--runner-label`.
- Do not add unnecessary exception handling in `main`; invalid config should
  fail directly.
- Use string matching/splitting for path metadata. Directory expansion may use
  filesystem APIs, but expanded paths are then handled as strings.
- Resource markers such as `one_card`, `two_node`, etc. are detected by checking
  whether the full path string contains the marker. Do not require the marker to
  be a path segment, and do not rely on directory ordering.
- Path expansion happens once in `main()` before routing. Framework functions
  receive one expanded path at a time and must not call `list_files()` or expand
  directories internally.
- Unrecognized schedule paths are collected as path strings and printed once as
  a single warning block with newline-separated entries.
- Keep the four test types as four functions: `single_node`, `multi_node`,
  `ops`, and `accuracy`. Avoid splitting their internal logic into many tiny
  helper functions.
- Do not hard-code a required prefix such as `tests/e2e/schedule/model` when
  classifying paths. Use string presence such as `model`, `ops`, `accuracy`,
  and resource keywords.
- Do not infer `external_dp` or `internal_dp` in the parser. Multi-node launch
  mode should be handled later in `run.sh`.
- Follow the structure in `parse_schedule_config_v2.md` lines 28-56:
  iterate over configured paths, try `single_node`, then `multi_node`, then
  `ops`, then `accuracy`, and collect unrecognized paths as warnings.
- The `single_node_matrix`-style names in the pseudocode mean "collect cases"
  first: use `single_node_list`, `multi_node_list`, `accuracy_list`, and
  `ops_list` to collect `PeriodicCase` objects.
- Generate the final workflow matrices only in post-processing, after case
  collection, filtering, and grouping.

Current implementation shape:

- `PeriodicCase` has:
  `name`, `config_path`, `test_content`, `framework`, `device_type`,
  `device_scale`, `device_num`, `runner`, `group`.
- Runner routing is internal via `runner_mapping_table`.
- The four route functions are:
  `single_node_framework`, `multi_node_framework`, `ops_framework`, and
  `accuracy_framework`.
- `main()` creates `single_node_matrix`, `multi_node_matrix`,
  `accuracy_matrix`, and `ops_matrix` only after collecting and filtering:
  `single_node_list`, `multi_node_list`, `accuracy_list`, and `ops_list`.
- Framework functions append `PeriodicCase` objects to those lists; they do not
  append final matrix dicts.
- Post-processing uses `build_single_node_matrix`, `build_multi_node_matrix`,
  `build_accuracy_matrix`, and `build_ops_matrix`.
- `multi_node_matrix` must not contain `multi_node_type`; `run.sh` owns that
  decision.
- `section["group"]` is the only schedule section identifier.
- CLI uses `--group`; `--schedule-name` and `--runner-label` have been removed.
- `selected_cases_summary` output has been removed.

Validation already run:

```bash
.venv/bin/python -m py_compile .github/workflows/scripts/parse_schedule_config.py
.venv/bin/ruff check .github/workflows/scripts/parse_schedule_config.py
.venv/bin/ruff format --check .github/workflows/scripts/parse_schedule_config.py
.venv/bin/python .github/workflows/scripts/parse_schedule_config.py \
  --config .github/workflows/scripts/schedule_config.yaml \
  --event-name workflow_dispatch \
  --group nightly-main \
  --test-filter Qwen3-235B-A22B-W8A8
```

The last command emits warnings for missing `tests/e2e/schedule/...` directories
because Step 5 has not migrated the tests yet. That is expected for now.

## Current Worktree Notes

Current branch: `ci_nightly`.

Known pre-existing/working files before this memory file:

```text
M  .github/workflows/_e2e_nightly_multi_node.yaml
M  .github/workflows/_e2e_nightly_single_node.yaml
M  .github/workflows/_e2e_nightly_single_node_models.yaml
?? .github/workflows/_e2e_periodic_ops.yaml
?? .github/workflows/scripts/parse_schedule_config.py
?? .github/workflows/scripts/parse_schedule_config_v2.md
?? .github/workflows/scripts/schedule_config.yaml
?? .github/workflows/scripts/schedule_periodic_test.yaml
```

Do not revert unrelated user changes.

## Next Action

Wait for user review/confirmation of Step 2. If confirmed, start Step 3 with a
`grill-me` question before editing workflow files.

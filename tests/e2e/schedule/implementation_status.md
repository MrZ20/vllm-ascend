# V2 implementation status

## Baseline and scope

- Execution references: original R1 and restored
  `vllm_ascend_schedule_v2_agent_execution_plan_revised.md`; see conversion notes
  for the architectural review and retained differences.
- Branch: `refactor_test_framework`, based on upstream main
  `0deca318102a07f9d8e027047652a2a0baab4fbc` through rebase commit `977d4c856`.
- Development is local; verification is delegated separately. No push or cron
  activation is part of this change.
- The six requested smoke cases use temporary YAML, two benchmark prompts and
  cached models. Results are workflow checks, not accuracy/performance claims.

## Work packages

| Package | State | Evidence / remaining validation |
| --- | --- | --- |
| P0 baseline | Complete | Repository/main verified; both Docker containers inspected; model caches and shared volume checked. |
| P1 schema | Implemented | Typed CPU-only parser, duplicate-key/unknown-field validation and config UT. |
| P2 planning | Implemented | Explicit/external expansion, mixed routing, devices/templates and pure planner UT. |
| P3 shared runtime | Implemented | Additive env/cwd/log/deferred readiness interfaces; old helper UT retained. |
| P4 launcher | Implemented | All peers start before wait; proxy dependencies, owned cleanup, lightweight real HTTP lifecycle UT. |
| P5 test handlers | Implemented | Existing request/AISBench tools in a bounded child; normal-service smoke passed on the old compatible engine. Other paths remain unverified. |
| P6 runner | Implemented | Resources, NPU preflight, per-case state protocol, finite waits, scoped signals, optional KV manager. |
| P7 CI entry | Implemented | Independent dispatch workflows, strict matrix resolver, source/version checks; no custom build manifest. |
| P8 LWS/artifacts | CPU-validated | Template/controller tests and actionlint. Real GitHub/LWS deployment NOT_RUN. |
| P9 documentation/smoke | Stopped at user request | Usage/conversion documents and a fresh-nightly migration/validation plan are ready. No further retry was started. |

## CPU validation

Commands use the local `.venv`:

```bash
.venv/bin/python -m pytest --noconftest tests/ut/schedule tests/ut/test_e2e_utils.py -q -rs
.venv/bin/ruff check tests/e2e/schedule tests/ut/schedule tests/e2e/utils.py tests/ut/test_e2e_utils.py
bash -n tests/e2e/schedule/scripts/run.sh
```

- Initial parser/planner: 39 passed.
- Initial combined suite: 110 passed, 3 skipped.
- CI suite addition: 121 passed, 3 skipped; actionlint issue corrected and verified.
- The local skips require Linux `/proc` or the absent local vLLM installation.
- Container shared-helper UT: 66 passed before model smoke.
- Revised-plan suite: **133 passed, 3 skipped** in 13.87 seconds, including both
  interrupted-node regression tests. Ruff check/format,
  all three V2 workflows (actionlint), bash syntax, default ShellCheck and all three
  Markdown files passed. Log: `/private/tmp/schedule-v2-final-related-ut.log`.

## Remote environment

- Direct SSH: `a3-node0`, `a3-node2`; container `zsl_nightly` on both.
- Addresses: `172.22.0.218`, `172.22.0.188`.
- Source root: `/vllm-workspace/vllm-ascend`; underlying source main `0deca3181`.
- Image tag: `swr.cn-north-12.myhuaweicloud.com/base_image/ascend-ci/vllm-ascend:nightly-ci-main-a3`.
- Installed environment observed: Python 3.12.13, vLLM `0.29.0+empty`, Ascend
  `0.19.1rc2.dev2294+gc173a64a4`; benchmark source `0da56eadb2ac85c31c2540f4f5b69af3ec5717a5`.
- The first normal-service attempt failed on missing `vllm.models.deepseek_v4_1`
  before readiness. It ran no benchmark and released its owned processes/devices.
- Follow-up validation uses `/tmp/schedule-v2-nightly-c173a64a4` at the installed
  Ascend commit `c173a64a44dec4ba97aaba6277b1dfc1562eda19`, with the framework
  overlay and existing native libraries. Original branch HEAD is preserved;
  this does not claim a native rebuild from the uncommitted local branch. Per-run framework/config digests identify test code.
- `/root/.cache` is the same shared SFS mount on both nodes. Temporary writes
  were read from the opposite node and then removed in both directions.
- `SCHEDULE_CLEANUP_PROCESSES=0` on every validation invocation. No global process
  kill is authorized. The chosen device pool is rechecked before every run.

| Priority / scenario | Allocation | State | Evidence |
| --- | --- | --- | --- |
| P0 normal Qwen32B | node0 12–15 | PASS on nightly-compatible engine | `smoke-single-20260921-02`: pytest 1 passed, health/chat 200, AISBench 2/2 success, owned cleanup passed; selected NPUs released. |
| P0 internal DP Qwen30B | both nodes 12,13 | FAIL during startup | `smoke-internal-20260921-04`: node2 Triton temporary-directory cleanup failed; no benchmark requests. Both launchers exited 1, cleanup passed, selected NPUs released. |
| P0 mixed PD Qwen30B | both nodes 12,13 | Pending | Explicit P + external D |
| P1 EPD Qwen2.5-VL | node0 12–15 | Pending | Encode + PD + proxy |
| P2 2P2D Qwen30B | node0 12–15 | Pending | Four TP1 engines + proxy |
| P2 external PD Qwen30B | both nodes 12,13 | Pending | Both sides external ranks |

Internal-DP attempt `01` did not start node2 because the remote launch command
had a shell-quoting error. Attempts `01` and `02` were then stopped during
validation-launcher coordination, before completing a two-node request run;
neither is an E2E pass.
Attempt `01` exposed a runner issue: an interrupted primary waited for an absent
peer after its own cleanup. The runner now publishes failure and returns without
that wait; both primary and worker paths have regression tests. Attempt `03` used the corrected runtime but exposed the primary rank CLI issue
described in the conversion notes. Attempt `04` corrected that CLI and reached
multirank initialization, then failed in node2 Triton temporary-directory cleanup.

Single-02 artifacts are under
`/root/.cache/schedule-v2-validation/smoke-single-20260921-02/` on the shared mount.
Its shutdown reports `resource_tracker` warnings for 124 semaphores and 6 shared
memory objects. Functional requests and NPU release passed; these warnings remain
unresolved and are not treated as a memory-management fix.

The user stopped further retries to rebuild the nightly image. No `05` attempt or
other topology was started. Both nodes were checked for owned process/port release;
NPUs 12 and 13 were idle. Evidence: `/private/tmp/schedule-v2-evidence/internal-04/`.

See [fresh nightly validation plan](validation_handoff.md) for the next agent's
image checks, selective migration and test sequence.

## Explicitly unverified outside the smoke acceptance

- Actual GitHub Actions/LWS scheduling, prepared image deployment,
  device plugin allocation, CPU/memory scheduling and Pod release in Kubernetes.
- Full accuracy/performance baselines and long-running stability.
- KV Pool hardware integration, speculative-acceptance metrics and all model
  variants/SoCs beyond the selected examples.

See [migration notes](migration.md) for every intentional plan adjustment and
C01–C24 mapping. These items are not reported as hardware passes.

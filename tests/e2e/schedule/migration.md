# V2 conversion and plan differences

## Selected examples

| Source | V2 file under `models/qwen/` | Deployment and test semantics |
| --- | --- | --- |
| Nightly `Qwen3-32B-Int8.yaml` | `Qwen3-32B-Int8.yaml` | One TP4 eager server, explicit full command and auto endpoint. Keeps the source accuracy/performance benchmark settings. Adds one short chat request. Does not copy the separate graph-capture variant. |
| Internal `DeepSeek-V3.1-BF16.yaml` topology | `Qwen3-30B-internal-dp.yaml` | Smaller Qwen30B W8A8 model; TP2 × DP2 across two nodes, one local rank per node, primary API plus headless worker. Two-request execution benchmark. |
| Internal `Qwen3-235B-disagg-pd.yaml` plus external `QWEN3_235B_PD.yaml` | `Qwen3-30B-mixed-pd.yaml` | One explicit TP2 prefiller and two external TP1 decoder ranks; independent Service coordinator/RPC address; PD proxy from Service endpoints. |
| Mixed V2 sample | `Qwen3-30B-external-pd.yaml` | Two TP1 external ranks in each of P and D, two cards per node. |
| New small topology example | `Qwen3-30B-2p2d.yaml` | Four explicit TP1 engines on one node, independent ports, two P and two D Service endpoints. |
| Weekly `Qwen2.5-VL-7B-Instruct-EPD.yaml` | `Qwen2.5-VL-7B-EPD.yaml` | Two engines (encode and combined prefill/decode) plus EPD proxy; empty prefill URL list becomes `disable`. TP2 per engine uses the four-card allocation. Keeps benchmark definitions and adds an image request. |

Cadence, node count and card count now live in the independent V2 matrices.
Model IDs, CLI configuration, declared environment, endpoints and benchmark
settings stay in the model YAML. Helpers do not infer routing roles from node
indices or filenames. Temporary remote smoke YAMLs use absolute cached model
paths, short contexts and two benchmark prompts; they are distinct from the
tracked full benchmark examples.

## Adjustments to Execution Baseline R1

1. **Benchmark launch adaptation.** The installed AISBench cannot match absolute
   model/dataset filenames through `--models/--datasets`. V2 combines per-run
   configuration files using AISBench's own config/type conversion and summarizer
   definitions, then runs a foreground CLI process. This retains existing result
   validation while bounding requests and descendant process lifetime.
2. **Bounded test child.** Request and benchmark helpers run in an owned child
   process because some existing helpers wait indefinitely internally. The parent
   enforces one finite test budget and monitors both local services and shared
   peer failures. There is no second server runtime.
3. **Prepared-image entry.** CI uses an already prepared compatible image and
   checks its source commit against the workflow ref. Package versions are
   recorded; a custom build attestation is not required. It does not build or
   reinstall packages inside allocated NPU Pods.
4. **Current runner labels.** Four-card direct entries use the existing
   `linux-aarch64-a3-800i-4` label found in the current nightly/weekly matrices;
   the plan's example nightly four-card label is not assumed to exist.
5. **KV Pool reuse location.** The proven multi-node manager remains in its
   existing module. A small optional typed pool/address interface serves V2
   without constructing a legacy topology config; existing V1 callers retain
   their signature and behavior. No new infrastructure DSL is introduced.
6. **Signal handling and shell logging.** The pytest entrypoint installs and
   restores a scoped SIGTERM handler. `run.sh` forwards termination to its own
   child PID and preserves both pytest and tee exit status. There is no active
   global server registry or global process sweep during normal cleanup.
7. **Small example scope.** Qwen32B's separate graph case is not duplicated;
   EPD uses TP2 for each engine to exercise the requested four-card allocation.
   The new topology examples check execution with two benchmark requests and
   have no asserted performance baseline. Existing large-model matrices and
   production cron remain on V1 until separate rollout validation.

## Review against the revised plan

The revised plan is an architectural reference, not a second implementation
mandate. The Service/rank model, global planning/local execution boundary and
common Remote* lifecycle are unchanged. The following choices are deliberate:

- Adopted optional `schema_version` and limited CI scaffolding to A3 nightly.
  EPD remains a model example and selectable nightly entry, without a separate
  weekly workflow. All entries are dispatch-only candidates pending CI validation.
- Removed the custom image build-manifest requirement. Exact CI source/version
  metadata is retained; native build correspondence is not inferred from Git.
- Kept bounded quote handling around Context/env substitution before `shlex`.
  Naive textual replacement corrupts quoted JSON, embedded quotes and backslashes;
  regression tests exercise these actual command arguments. No general template
  evaluator or shell execution is introduced.
- Kept optional shared per-case coordination. Multiple cases can reuse the same
  ports, so workers must finish cleanup before the next case starts. HTTP endpoint
  disappearance alone does not convey a peer's failed cleanup. Single-case local
  execution can still use ready/unready without a shared directory. Failure and
  termination publish the local failure after cleanup and return immediately;
  they do not wait for nodes that never started.
- Kept single-node automatic ports and plan/effective JSON diagnostics. Both are
  already small, tested helpers; multinode still uses fixed ports. The two JSON
  files aid device/env debugging and are not separate mandatory schemas.
- One-server and multi-server plans use the same prepared `RemoteServerGroup`
  path. A separate single-server branch would duplicate allocation and readiness
  logic; the shared Remote runtime remains the only process owner.
- Device expansion uses the actual visible IDs (including non-contiguous pools),
  rather than assuming zero-based physical cards. The common allocator retains
  upstream prefill-context parallelism as well as TP, PP and local DP.
- Cleanup defaults to off even though the revised plan allows isolated-container
  startup sweeps. This follows the user's explicit privileged-container constraint.

The real engine import mismatch was reproduced on main `0deca3181`. Hardware
validation therefore uses a separate worktree at the installed nightly Ascend
commit `c173a64a44dec4ba97aaba6277b1dfc1562eda19`, with only framework changes
overlaid. The original remote branch is preserved. This validates the framework
against that compatible runtime, not current main's engine implementation.

The real Internal DP launch also corrected a sample CLI detail: on this vLLM,
`--data-parallel-start-rank 0` on a non-headless primary implies hybrid/external
load balancing. The primary now omits it and uses the default rank zero; the
headless worker explicitly retains `--data-parallel-start-rank 1`. The planner
continues to pass the declared command unchanged.

## C01–C24 implementation trace

| Item | Implementation / evidence |
| --- | --- |
| C01 | `Launcher` uses existing `_ManagedProcess`, `RemoteServerGroup`, `RemoteProxy`; shared-helper regression UT. |
| C02 | Independent V2 matrices/resolver/workflows; `test_v2_workflows_are_independent_dispatch_only`. |
| C03 | `ResourceSpec`, `resolve_cluster`, matrix resource validation UT. |
| C04 | Direct allocation detection and explicit LWS default; omission/zero tests in `test_ci.py`. |
| C05 | Separate `NodeSpec`, `ClusterContext`, `NodeContext`. |
| C06 | CPU-only records/immutable Context registry in `config.py`; isolated import test. |
| C07 | Service coordinator selection; mixed-topology UT including a decoder coordinator on node1. |
| C08 | Declared env preserved; separate plan/effective records and per-server env UT. |
| C09 | Protected quote-aware substitutions; whitespace/quotes/JSON/literal-dollar and shell rejection UT. |
| C10 | Inherited env plus declared dependency resolution; missing/cyclic/self-reference tests. |
| C11 | Single allocation for `port: auto`; reuse/duplicate/multinode-auto tests. |
| C12 | Remote endpoint dependency isolation; remote inherited/local NIC rejection tests. |
| C13 | Case-wide reservation before default allocation; later-explicit-device regression UT. |
| C14 | Prepare/start all peers before wait; real lightweight HTTP peer/proxy lifecycle UT. |
| C15 | Constructor cleanup, reverse owner shutdown, sibling cleanup collection and preserved errors; failure UT. |
| C16 | Separate completion/tokenize/metrics targets, clean exit detection, local-stop distinction. |
| C17 | `Coordinator` ready/stop/node-result/final protocol; two-thread cleanup/failure/identity/missing-node UT. |
| C18 | Cleanup opt-in defaults to zero; owned process groups and PID-isolated LWS template. |
| C19 | Short NPU query child, all-selected-card free-memory check, bounded release wait. |
| C20 | LWS `leaderWorkerTemplate.size` retained; rendered template UT. |
| C21 | Required run-scoped host anti-affinity, shared PVC and controller host/result checks. Remote shared Docker volume probed across nodes. |
| C22 | Source/version metadata, bounded exit and artifact collection. CI deployment remains separately gated by real validation. |
| C23 | Common typed KV config plus existing manager's optional pure inputs; old/new interface UT. KV Pool hardware scenario is outside this smoke matrix. |
| C24 | Explicit handler set, backend tokenization/metrics, existing request/benchmark checks; unknown content rejected. Accuracy/performance measurements are not part of smoke acceptance. |

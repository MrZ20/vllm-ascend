# Scheduled E2E V2

V2 uses one model schema and one entrypoint for ordinary serving, internal DP,
external DP ranks, PD and EPD. Resources come from the allocation or CI matrix.
V1 remains available at its existing entrypoints.

## Write a model configuration

```yaml
name: example
model: Qwen/Qwen3-0.6B
server_cmd: >-
  vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0
  --port {{ LaunchContext.port }} --tensor-parallel-size 1
endpoint: {host: '{{ NodeContext.ip }}', port: auto}
test_content: [chat_completion]
prompts: ['Hello!']
api_keyword_args: {max_tokens: 16}
```

`server_cmd` is a complete command beginning with `vllm serve MODEL`. Use
`deployment.services` for multiple launch locations; do not combine both forms.
Each Service contains `launch_strategy` and a `launches` mapping keyed by
`node0`, `node1`, etc. See [model examples](models/qwen).

- `explicit`: one command per listed node. Declare its serving `endpoint`, or
  `null` for a headless participant. The YAML supplies DP/headless/KV CLI flags.
- `external_dp_rank`: declare `expansion` with `dp_size`, `dp_size_local`,
  `dp_rank_start`, `port_start`, `dp_rpc_port`, and `tp_size`. Optional `pp_size`,
  `cp_size`, `sp_size` default to one. One process is expanded per local rank.
  Its device count is TP × PP × CP × SP. Optional `device_ids` names its pool.
- A Service's coordinator defaults to its lowest node index. An explicit
  `coordinator_node` must belong to that Service.
- `routing.groups` contains Service names. PD uses `prefiller`/`decoder`; EPD
  uses `encode`/`prefill`/`decode`. Empty EPD `prefill` is supported.
- Without a proxy there must be exactly one global serving endpoint. With a
  proxy, its command and endpoint are explicit, and it starts after its backends
  become ready. All local peer servers start before readiness is awaited.

`schema_version` may be omitted; if supplied it must be `2`.
Native YAML anchors and merge keys are supported. Duplicate explicit keys,
unknown schema fields and unsupported test handlers are rejected.

### Templates and environment

The Context definitions and public-field registry are in [config.py](config.py).

| Context | Frequently used fields |
| --- | --- |
| `NodeContext` | `id`, `index`, `ip`, `npu_per_node`, `nic_name` |
| `ServiceContext` | `name`, `coordinator_node_id`, `coordinator_ip` |
| `LaunchContext` | `port`, `local_rank`, `dp_rank`, `visible_devices` |
| `PDRoutingContext` | `prefiller_hosts`, `prefiller_ports`, `decoder_hosts`, `decoder_ports` |
| `EPDRoutingContext` | `encode_urls`, `prefill_urls`, `decode_urls` |

`nic_name` is available only for the local node. Remote endpoints may use global
Contexts and declared environment constants, not a remote process's inherited
environment. Multinode endpoints require fixed, reachable addresses/ports.
Single-node `endpoint.port: auto` chooses one port reused throughout that launch.

`$VAR` and `${VAR}` may reference declared or inherited environment values.
Declared env dependencies are resolved by name; self-reference extends an
inherited value, for example `LD_LIBRARY_PATH: /extra:$LD_LIBRARY_PATH`.
Substituted text is never expanded again. In commands, quoted substitutions
remain one argument; a standalone unquoted substitution can expand a whitespace
list. Single-quoted `$VAR` and escaped dollar signs remain literal. Shell pipes,
redirection and command substitution are unsupported.

Use a quoted env substitution for dynamic JSON:

```yaml
envs:
  EC_CONFIG: '{"ec_connector_extra_config":{"shared_storage_path":"/dev/shm/$RUN_ID"}}'
server_cmd: >-
  vllm serve MODEL --ec-transfer-config "$EC_CONFIG"
```

All explicitly assigned device pools are reserved before any implicit allocation.
Per-launch env values override inherited/common defaults. The common server
allocator also counts pipeline and prefill-context parallelism. Selected IDs are
physical visible IDs; the short NPU probe queries their remapped logical indices.

### Tests and dependencies

Supported `test_content`: `completion`, `chat_completion`, `image`,
`spec_decode_acceptance`, `benchmark_comparisons`. Existing request and AISBench
helpers supply response/baseline checks. PD tokenization uses a prefiller;
metrics use a decoder; requests use the proxy. Benchmark comparison currently
supports TTFT with `<`, `<=`, `>` or `>=`.

For an image request, provide `mm_request.images` (use `[null]` for the cached
default image) or explicit image parts in `mm_request.messages`. `image_path`
selects the default file; it does not itself add an image to the request.

Benchmark input keys match `tools/aisbench.py`. Set `dataset_path_local` and
`model_path` to existing local caches for offline runs. The benchmark source
configuration tree must exist at `$BENCHMARK_HOME` (default `$REPO_ROOT/benchmark`).
Every benchmark gets a separate output directory; generated configs are copied
into the run, and the AISBench process belongs to the bounded test process group.
A zero baseline in the topology smoke examples means they check execution only.

`special_dependencies` maps package names to exact installed versions; installation
belongs to image preparation. The common `kv_pool` Mooncake/Memcache configuration
is supported using the existing manager's typed pool/address inputs. Its config
files and owned service logs are isolated by run and case. It is optional and is
not implicitly inferred from KV transfer flags.

## Run locally

Use a prepared environment with vLLM, Ascend, the requested models, and AISBench
installed. `run.sh` does not checkout, build or install packages.

```bash
export REPO_ROOT=/vllm-workspace/vllm-ascend
export CONFIG_YAML_PATH=tests/e2e/schedule/models/qwen/Qwen3-30B-internal-dp.yaml
export ASCEND_RT_VISIBLE_DEVICES=12,13
export CLUSTER_HOSTS=172.22.0.218,172.22.0.188
export LWS_WORKER_INDEX=0  # set 1 on the second node
export RUN_ID=my-unique-run
export LOG_PREFIX=/root/.cache/schedule-v2
export COORD_DIR=/root/.cache/schedule-v2
export SCHEDULE_CLEANUP_PROCESSES=0
bash "$REPO_ROOT/tests/e2e/schedule/scripts/run.sh"
```

Run on both nodes with the same YAML, `RUN_ID`, hosts and shared directories.
Recheck free cards before selecting them. The preflight uses a separate short
Python process, checks every selected NPU, and waits at most 30 seconds for at
least 90% free memory. This is an observation, not a reservation or OOM guarantee.

`NUM_NODES` defaults from hosts/LWS, then one. `NPU_PER_NODE` defaults to the
actual visible allocation. Conflicting resource inputs fail. Single-node runs
need neither hosts nor worker index. An explicit NPU count must match the
selected visible pool; it does not select the first N cards automatically.

With `COORD_DIR`, ready/stop/node-result/final files coordinate each case. Final
success requires every node's cleanup result. Multinode files containing multiple
cases require this shared protocol. Without it, a single-case worker only reports
`local_stopped`; disappearance of the primary is not proof of global success.

Startup cleanup defaults to **disabled**. `SCHEDULE_CLEANUP_PROCESSES=1` is only
for dedicated, PID-isolated disposable containers. Keep it `0` in the supplied
privileged validation containers. Normal completion and failures stop only the
process groups owned by this run. Scoped termination forwarding reaches the
pytest cleanup handler.

## CI

The independent A3 entrypoint is `schedule_nightly_test_a3_v2.yaml`. It is
manual-dispatch only, with no cron. Select comma-separated matrix names or `all`;
the workflow ref is the code/config ref under test. Weekly and other SoCs remain
outside this first rollout.

Supply a prepared image containing that source commit and compatible vLLM,
Ascend native extensions and benchmark dependencies. A digest-pinned image is
recommended for reproducibility; an extra build-manifest format is not required.
The entrypoint checks the source SHA and tracked tree, then records installed
package versions and configuration/framework digests. These checks identify the
test environment; they do not prove that native libraries were built from a
later checkout. Image preparation remains separate from allocated NPU runs.

The matrix is `SoC -> single_node/double_node/multi_node -> test_config`.
Entries have `name`, repo-relative `config_path`, `num_nodes`, optional
`npu_per_node`, and `os` for direct runners. Omitted direct-runner NPU count uses
the visible allocation; omitted LWS count uses its explicit platform default.
An explicit zero is rejected. Current entries request 4 cards locally or 2 per
node. A3 execution order/concurrency remains multi (2), double (4), single (9).

LWS uses one replica, CRD `leaderWorkerTemplate.size`, matching NPU requests and
limits, required per-run host anti-affinity, and the existing shared PVC.
CPU/memory requests are centralized reusable-workflow inputs (16 CPU/128 GiB),
not old full-host reservations. The controller must see the same shared storage
mounted at `/root/.cache`; missing coordination/results fails the run.

The controller checks actual Pod host placement, terminated exit codes,
restarts/OOM, all case results and required test artifacts. It handles containers
that finish before first observation. Diagnostics and owned-LWS deletion run
on failure; deletion is restricted to this run's name/label.

## Logs and troubleshooting

Under `$LOG_PREFIX/$RUN_ID`:

- `node-N/run_metadata.json`: versions, resources, source and config digests.
- `node-N/pytest.log` and `npu_preflight.json`.
- `node-N/cases/NNN-name/launch_plan.json` and `launch_effective.json`.
- Per-process logs, `tests.log`, per-benchmark outputs, `cleanup.json`, `result.json`.
- `cases/NNN-name/`: shared state when `COORD_DIR` equals `LOG_PREFIX`.
- `benchmark_results/NNN-name.json`: test/benchmark results on primary.
- `infrastructure/NNN-name/`: optional pool service/config files.

Start with the case `result.json` phase, then its server or test log. A process
exiting with code zero during startup/tests is still a failure. `stop` means the
test phase ended; only `final` plus successful node results proves cluster success.

See [implementation status](implementation_status.md) for tested scope and
[conversion notes](migration.md) for changes from the execution plan. The next
validation round starts from a newly built image using the
[fresh nightly migration and test plan](validation_handoff.md).

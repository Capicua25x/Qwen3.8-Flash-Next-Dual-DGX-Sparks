# FP8 PLE offload on 2x DGX Spark — findings (2026-09-14)

Working branch: `ple-offload-fp8`. Baseline: FP8 lane, `PLE_OFFLOAD=false`,
`SKIP_PLE_PATCH=true`, no offload wiring in `start.sh`.

## What we built

- `files/build_ple_packed_table_fp8.py` — builds the packed table for an FP8
  checkpoint. Pure byte concatenation of the 128 equal-sized `F8_E4M3` shards
  (row = head_dim bytes, no scales to interleave); the NVFP4 sibling packs
  `cat(codes, scales)` into 90-byte rows. Stdlib only (the host has no numpy).
  Verified: 320,001,536 rows x 160 B = 47.68 GiB, built in 66 s.
- `start.sh` step 6c — applies `patch_ple_offload.py`, bind-mounts the patched
  worker/connector on both nodes, syncs + mounts the packed table, sets
  `VLLM_PLE_PACKED_TABLE_DIR`.
- `start-fp8.sh` — `SKIP_PLE_PATCH` made overridable (was hard `true`).

## What already existed (and was never wired)

The offload machinery is complete in `files/` but the TP2 `start.sh` never
invoked it: `patch_ple_offload.py` runs clean (all anchors unique), the FP8
output dtype is implemented in `worker.py::get_offload_output_dtype`, and
`_attach_packed_table` skips the width check for FP8 (no `packed_row_width` on
`Qwen3_8FlashNextPLEFp8EmbeddingMethod`). The TP1 lane (`tp1/start.sh`) wires
all of it properly, including a cgroup cap + `memwatch.sh` watchdog, because
PLE_OFFLOAD is mandatory there.

## First blocker

`vllm/v1/worker/gpu_worker.py::_validate_ple_offload_config` rejects
`parallel_config.nnodes != 1`:

    ValueError: VLLM_PLE_CPU_OFFLOAD does not support the requested
    configuration. Unsupported settings: nnodes=2

It is the ONLY failing check for our TP2 config (DP=mp, PP=1, PCP=1, DCP=1,
no ubatching, architecture allowed, no weight transfer).

## First multi-node boot: the FP8 format works

With the guard lifted (`VLLM_PLE_OFFLOAD_ALLOW_MULTINODE=1`, commit `ee12cb8`)
the pair booted past it. Proven working:

- FP8 packed table built (47.68 GiB, byte-exact) and visible in both containers.
- The offload worker logs, verbatim:

      PLE ...: using packed mmap table /var/tmp/qwen38-ple-packed/...packed_u8
      PLE ...: mmap table attached (320001536 rows x 160 B = 47.68 GiB)
      PLE weight loading complete.
      PleOffload: registered 1 PleOffloadLayer(s) (dp_rank=0, tp_rank=0, ipc_addr=ipc:///tmp/...)

So the FP8 packed format, the mmap attach and the FP8 output dtype are all
correct — the new builder works.

## The real blocker (corrected): the worker is node-local by design, the code around it is not

The first boot's failure was read as "two node-local workers, each waiting for
2 registrations". Reading `spawn_ple_offload`, `multiproc_executor.py` and
`accept_registrations` together shows something simpler: only ONE offload
worker ever existed.

- `spawn_ple_offload` spawns on GLOBAL rank 0 only (`self.rank != 0`).
- `multiproc_executor.py` gives node 1's only rank global rank 1
  (`global_start_rank = local_world_size * node_rank_within_dp`), so node 1
  never spawned a worker.
- `parallel_config._ple_offload_ipc_path` is generated per config
  (`get_open_zmq_ipc_path()` -> `ipc://<base>/<uuid4()>`), so node 1's
  connector connected to a path with no listener. A zmq PUSH `connect()` to a
  missing ipc path succeeds; the registration queues in the socket and is
  dropped at close (`linger=0`) with no error. The `PleOffload: registered
  ...` lines on both nodes are the CONNECTORS logging — they are not proof of
  two workers, and the two different ipc addresses are the two configs' paths,
  not two bound sockets.
- The one worker (node 0) waited for `num_workers = dp_size * tp_size = 2`
  registrations, received only rank 0's, and timed out:

      TimeoutError: PLE offload worker did not become ready within 600.0s

So the guard does encode a real invariant, but the invariant is finer than
"single node": the whole protocol assumes ONE process tree — a single worker
for all ranks, spawned from global rank 0, counted world-wide, with a single
request sender per DP group. Multi-node needs the per-node worker topology
that the worker-side docstrings already describe ("serve every local DP rank",
"one CPU offload process for all local DP and TP workers") but that the
spawn/accounting/request paths never implemented.

Three gaps, not one:

1. **spawn** — global rank 0 (`self.rank != 0`) instead of each node's first
   rank;
2. **accounting** — `num_workers = dp_size * tp_size` (world) and validation
   against global TP rank sets, instead of the node's local rank count and
   local rank sets;
3. **requests/inputs** — `_launch` sends only from `tp_rank == 0`, and
   `_pin_input_buffers` only runs there. On a second node the local rank would
   wait forever on a done flag nobody could set, and its worker would never
   receive a request — gaps 1+2 alone still deadlock.

## The fix on this branch

One offload worker per node, serving that node's local ranks:

- `spawn_ple_offload` spawns from each node's first DP0 rank
  (`rank == node_rank_within_dp * local_world_size`);
- `num_workers = local_world_size`;
- every registration carries its global `rank`; the worker validates that the
  received rank set equals the node's local rank range, rejects duplicate
  `(dp_rank, tp_rank)` slots, and picks the lowest-rank local member of each DP
  group as the input/request leader;
- the connector computes `is_local_leader` (lowest local rank of its DP group)
  and only the leader pins/stages input buffers and sends
  `PleOffloadRequest`. Every rank still blocks on its own done flag; inputs are
  TP-replicated, so the local copy is equivalent.

Single-node behaviour is unchanged: the leader is TP0 of each DP group and
`local_world_size == dp_size * tp_size`, so the registration set, the leader
and the request sender are exactly what they were.

The alternative — one worker serving both nodes over the fabric — is not
available today: CUDA IPC output buffers and file_system shared memory never
cross nodes, so it would need a different transport for both. Per-node
workers duplicate the CPU forward once per node (the table is already mmapped
locally on both); that is cheap next to the GPU forward and is the only
transport-correct option.

## Validation status

NOT YET BOOT-VALIDATED — the pair was restored to DS4 (`deepseek-v4-flash`)
right after this autopsy. Planned checks on the next FP8 window:

- each node logs `Bound IPC address ...; waiting for 1 GPU worker
  registration(s)` and `Registrations complete`;
- boot reaches `:8888` with the packed table attached on both nodes;
- long-context sweep + KV pool size vs the FP8 baseline;
- single-node regression (nnodes=1) still serves.

**Safety rails worked:** cgroup cap held at 40.0 GiB; memwatch logged
`avail=44948MiB ... container=40956MiB` throughout; host MemAvailable stayed
~44 GiB. No host hang.

**Also fixed during the first boot (committed):**
- `HEAD_PLE_OFFLOAD_MOUNTS` was clobbered by a later `=` after the `+=`
  (gpu_worker mount lost) — reordered.
- The packed-table mount was added to the dead `DOCKER_ARGS` path instead of
  the live heredoc mount variables — moved to `HEAD/WORKER_PLE_OFFLOAD_MOUNTS`.

---

## UPDATE — validation boot (2026-09-14, same day)

The fix was booted on the pair (`PLE_OFFLOAD=true`, GPU_MEMORY_UTILIZATION=0.70,
TP2/2 nodes). Results:

**Topology fix: WORKS.**

- Both nodes spawned their own offload worker and served exactly one local
  registration:
  - gx10a: `GPU worker 0 registered (dp_rank=0, tp_rank=0)` →
    `Registrations complete` → `Busy-loop started`;
  - gx10b: `GPU worker 1 registered (dp_rank=0, tp_rank=1)` →
    `Registrations complete` → `Busy-loop started` (this node never had a
    worker before the fix).
- Boot reached `:8888` (`qwen3.8-flash-next-fp8`, max_model_len 262144),
  KV cache 1.58M tokens (6.05x concurrency at 262k).

**First real forward crashed — a second, unrelated bug found and fixed
(`96aa55d`).** The very first CPU forward on both nodes hit
`RuntimeError: index_select(): self and result must have the same scalar
type` in `ple_layer.py::forward_impl`'s packed-table branch: the mmapped
table is uint8 bytes, while the FP8 output buffer is float8_e4m3fn by design
(the GPU side bit-views the rows and dequantizes with the retained scale),
so `index_select(..., out=buffer)` is illegal. NVFP4 buffers are uint8 and
were unaffected. Fix: keep the zero-copy `out=` for uint8 buffers and
gather-then-bit-view (`rows.view(output.dtype)`) for float8 buffers. This
path had never executed anywhere before (previous boots never got past the
registration deadlock and the TP1 lane ships NVFP4).

**After the fix:** real generations served with zero worker errors on both
nodes across two requests (57 and 93 completion tokens); single-stream speed
in line with the ~47 tok/s thinking-on baseline.

**Multi-node DP gate (same commit):** `_validate_ple_offload_config` now
rejects `nnodes>1 && DP>1` with a clear message (a node-local worker only
serves one dp0 replica per node; that config previously died with a bare
`IndexError` in the connector).

## Sweep results (thinking on, EP A/B + chunk A/B; 2 rounds, levels 1 and 6)

| config             | mix c1 agg / TTFT | mix c6 agg / TTFT | long c1 agg / prefill | long c6 agg / TTFT |
|--------------------|-------------------|-------------------|-----------------------|--------------------|
| EP-on  chunk 4096  | 33.7 / 4.14 s     | 69.5 / 12.74 s    | 10.9 / 2196 t/s       | 14.8 / 11.39 s     |
| EP-off chunk 4096  | 36.2 / 4.10 s     | 75.8 / 11.38 s    | 11.8 / 2293 t/s       | 15.4 / 10.99 s     |
| EP-off chunk 8192  | 35.4 / 3.20 s     | 73.8 / 14.76 s    | 12.6 / 2425 t/s       | 16.6 / 10.70 s     |

- **EP-off is adopted** — it wins every cell (agg +7-9%), contrary to the
  earlier "−4% at TP2" note.
- **Chunk 8192 is kept for the long-context lane**: the flatlining long shape
  improves everywhere (+6-8% agg, TTFT, prefill +5.8%), at the cost of mix c6
  (−2 agg, +3.4 s TTFT). Tony's +59% prefill claim does not reproduce
  (+5.8%). Two rounds is thin; re-confirm with a 3-round, 4-level run before
  treating either chunk choice as final.
- `.env` now: `ENABLE_EXPERT_PARALLEL=false`, `MAX_NUM_BATCHED_TOKENS=8192`,
  `GPU_MEMORY_UTILIZATION=0.70`, `PLE_OFFLOAD=true` (deliberately
  uncommitted).

## Status

- Branch `ple-offload-fp8` on gx10a: `24583f3` (topology fix) + `96aa55d`
  (dtype fix + DP gate). **Not pushed anywhere yet.**
- PR #54 (draft) to MiaAI-Lab remains as-is pending the operator's go to
  push the validated commits.

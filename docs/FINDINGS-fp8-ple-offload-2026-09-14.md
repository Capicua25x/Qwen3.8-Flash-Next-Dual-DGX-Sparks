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

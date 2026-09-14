# FP8 PLE offload on 2x DGX Spark — findings (2026-09-14)

Working branch: `ple-offload-fp8` (commit cccd2ff). Baseline: FP8 lane,
`PLE_OFFLOAD=false`, `SKIP_PLE_PATCH=true`, no offload wiring in `start.sh`.

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

## The blocker

`vllm/v1/worker/gpu_worker.py::_validate_ple_offload_config` rejects
`parallel_config.nnodes != 1`:

    ValueError: VLLM_PLE_CPU_OFFLOAD does not support the requested
    configuration. Unsupported settings: nnodes=2

It is the ONLY failing check for our TP2 config (DP=mp, PP=1, PCP=1, DCP=1,
no ubatching, architecture allowed, no weight transfer).

## Why the guard is too strict for this architecture

The offload path is node-local by construction:
- each GPU worker spawns its own offload process (`spawn_ple_offload`, a local
  `multiprocessing.Process`, not a distributed process group);
- coordination is zmq over `parallel_config._ple_offload_ipc_path`, a per-node
  local address;
- shared memory uses `torch_mp.set_sharing_strategy("file_system")` (local shm).

The only `world_size` use is an identity scalar (`worker_id = dp_rank *
world_size + rank`). At nnodes=2 there are two independent node-local offload
workers, each mmapping its own local copy of the packed table — which is
exactly why the table must exist on both nodes.

## Why TP1 cannot be used to isolate the test

FP8 weights are 1 byte/param: 172.8 GiB on disk, of which 47.68 GiB is the PLE
table -> ~125.1 GiB of non-PLE weight. One Spark is a ~121.7 GiB unified pool
with ~5.6 GiB runtime overhead, so full FP8 does not fit at TP1 (this is the
same reason the TP1 lane ships the smaller NVFP4 checkpoint). TP2 (~62.5 GiB
of weight per node) is the only configuration where FP8 fits — so the nnodes
guard must be patched; there is no single-node isolation path.

## Next step

Overlay `gpu_worker.py` (same patch mechanism as `patch_ple_offload.py`) to
allow `nnodes>1` for the node-local offload design, relaunch TP2, and compare
the long-context sweep (the shape that flatlines: 11.1 tok/s c1, TTFT climbing
linearly to 11.3 s at c6) plus the KV pool size.

---

## UPDATE — first multi-node boot (2026-09-14, later)

The nnodes guard was patched (one hunk, `VLLM_PLE_OFFLOAD_ALLOW_MULTINODE=1`)
and the pair booted past it. Results:

**Proven working:**
- FP8 packed table built (47.68 GiB, byte-exact) and visible in both containers.
- The offload worker logs, verbatim:
  `PLE ...: using packed mmap table /var/tmp/qwen38-ple-packed/...packed_u8`
  `PLE ...: mmap table attached (320001536 rows x 160 B = 47.68 GiB)`
  `PLE weight loading complete.`
  `PleOffload: registered 1 PleOffloadLayer(s) (dp_rank=0, tp_rank=0, ipc_addr=ipc:///tmp/...)`
- So the FP8 packed format, the mmap attach and the FP8 output dtype are all
  correct — the new builder works.

**New blocker — the real reason the nnodes guard exists.**
Both nodes registered, but to DIFFERENT ipc addresses (each node spawns its own
offload worker with its own `ipc:///tmp/<uuid>`):
  gx10a tp_rank=0 -> ipc:///tmp/be137422-...
  gx10b tp_rank=1 -> ipc:///tmp/9abeecbb-...
Then:
  TimeoutError: PLE offload worker did not become ready within 600.0s

`PleOffloadWorker.accept_registrations` expects `num_workers = dp_size *
tp_size` (=2 at TP2) registrations in a SINGLE worker process. Each nodes

---

## UPDATE — first multi-node boot (2026-09-14, later)

The nnodes guard was patched (one hunk, `VLLM_PLE_OFFLOAD_ALLOW_MULTINODE=1`)
and the pair booted past it. Results:

**Proven working:**
- FP8 packed table built (47.68 GiB, byte-exact) and visible in both containers.
- The offload worker logs, verbatim:
  - `PLE ...: using packed mmap table /var/tmp/qwen38-ple-packed/...packed_u8`
  - `PLE ...: mmap table attached (320001536 rows x 160 B = 47.68 GiB)`
  - `PLE weight loading complete.`
  - `PleOffload: registered 1 PleOffloadLayer(s) (dp_rank=0, tp_rank=0, ipc_addr=ipc:///tmp/...)`
- So the FP8 packed format, the mmap attach and the FP8 output dtype are all
  correct — the new builder works.

**New blocker — the real reason the nnodes guard exists.**
Both nodes registered, but to DIFFERENT ipc addresses (each node spawns its own
offload worker with its own `ipc:///tmp/<uuid>`):

    gx10a tp_rank=0 -> ipc:///tmp/be137422-...
    gx10b tp_rank=1 -> ipc:///tmp/9abeecbb-...

Then:

    TimeoutError: PLE offload worker did not become ready within 600.0s

`PleOffloadWorker.accept_registrations` expects `num_workers = dp_size *
tp_size` (=2 at TP2) registrations in a SINGLE worker process. Each node's
worker therefore waits for 2 registrations but only ever receives its own
node's (1). Two workers, each waiting for two, each getting one -> deadlock.

So the guard is NOT merely conservative: it encodes the invariant that ONE
offload worker serves all TP ranks, which holds only when they share a process
tree (single node). The earlier "node-local, therefore safe" reading was right
about the mechanism but wrong about the accounting — the worker counts
registrations globally.

**What a real multi-node fix needs** (pick one):
1. Per-node offload workers, each with `num_workers = 1` (the local TP rank),
   and the registration/target bookkeeping keyed by local rank. This is the
   node-local design the architecture already implies — the largest but most
   correct change.
2. A cross-node rendezvous so one worker serves both ranks over the fabric
   (what the guard/design assumes) — much harder: IPC output buffers are CUDA
   IPC handles and the request path is a local zmq PUSH, neither of which
   crosses nodes today.

Option 1 matches what each node already does (spawn local worker, mmap local
table) and is the recommended path.

**Safety rails worked:** cgroup cap held at 40.0 GiB; memwatch logged
`avail=44948MiB ... container=40956MiB` throughout; host MemAvailable stayed
~44 GiB. No host hang.

**Also fixed during this session (launcher bugs, committed):**
- `HEAD_PLE_OFFLOAD_MOUNTS` was clobbered by a later `=` after the `+=`
  (gpu_worker mount lost) — reordered.
- The packed-table mount was added to the dead `DOCKER_ARGS` path instead of
  the live heredoc mount variables — moved to `HEAD/WORKER_PLE_OFFLOAD_MOUNTS`.

#!/usr/bin/env python3
"""Allow VLLM_PLE_CPU_OFFLOAD with nnodes>1 (multi-node TP).

Why this exists
---------------
vLLM's stock guard `gpu_worker.py::_validate_ple_offload_config` rejects any
PLE CPU-offload config with `parallel_config.nnodes != 1`:

    ValueError: VLLM_PLE_CPU_OFFLOAD does not support the requested
    configuration. Unsupported settings: nnodes=2

That blanket rejection is conservative, not structural. The offload path is
node-local by construction:

  * each GPU worker spawns its OWN offload process
    (`gpu_worker.py::spawn_ple_offload` -> a local multiprocessing.Process,
    not a distributed process group);
  * coordination is zmq over `parallel_config._ple_offload_ipc_path`, a
    per-node local address;
  * shared memory uses torch_mp sharing strategy "file_system" (local shm).

The only use of `world_size` in the connector is an identity scalar
(`worker_id = dp_rank * world_size + rank`). At nnodes=2 there are simply two
independent, node-local offload workers, each mmapping its own local copy of
the pre-packed table (which is why the table must exist on both nodes).

This is what makes FP8 viable on 2x DGX Spark at all: FP8 weights are
1 byte/param (~125.1 GiB of non-PLE weight), which does not fit one Spark
(~121.7 GiB unified pool), so TP2 is the only configuration in which full FP8
fits -- and it needs the offload to keep the 47.7 GiB PLE table out of UVM.

The edit is deliberately minimal: the nnodes condition is AND-ed with an env
escape hatch (VLLM_PLE_OFFLOAD_ALLOW_MULTINODE=1). Default behaviour is
unchanged; every other check in the guard stays exactly as upstream.

Inputs:  files/gpu_worker/gpu_worker.py.orig (extracted from the image)
Outputs: files/gpu_worker/gpu_worker.py      (bind-mounted over the package)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "gpu_worker", "gpu_worker.py.orig")
OUT = os.path.join(HERE, "gpu_worker", "gpu_worker.py")

OLD = """        if parallel_config.nnodes != 1:
            unsupported.append(f"nnodes={parallel_config.nnodes}")"""

NEW = """        # --- APEXiA 2026-09-14: opt-in multi-node escape hatch ---
        # The offload path is node-local (each GPU worker spawns its own
        # offload process; per-node zmq ipc addr; local file_system shm), so
        # nnodes>1 means N independent node-local offload workers, not a
        # cross-node coordination requirement. Gated on an explicit env flag
        # so the upstream default is unchanged.
        if parallel_config.nnodes != 1 and os.environ.get(
            "VLLM_PLE_OFFLOAD_ALLOW_MULTINODE", "0"
        ) != "1":
            unsupported.append(f"nnodes={parallel_config.nnodes}")"""

src = open(ORIG).read()
count = src.count(OLD)
if count != 1:
    raise SystemExit(f"gpu_worker.py: anchor not unique/missing (count={count})")
if "\nimport os\n" not in src and not src.startswith("import os\n"):
    raise SystemExit("gpu_worker.py: no top-level `import os` -- adjust the patch")
src = src.replace(OLD, NEW, 1)
open(OUT, "w").write(src)
print("patched gpu_worker.py")

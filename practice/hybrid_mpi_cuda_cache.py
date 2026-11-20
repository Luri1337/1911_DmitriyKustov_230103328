"""Hybrid MPI-CUDA stencil and MPI-3 shared-memory false-sharing lab.
Run with: mpirun -np 4 python hybrid_mpi_cuda_cache.py GITHUB_USERNAME
Designed for a single-node, four-rank Google Colab runtime with an NVIDIA T4.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64, types
from numba.extending import intrinsic

# ---------------------------- Experiment constants ---------------------------
TILE = 16
SHARED = TILE + 2
NX = NY = 512
LOCAL_NX = LOCAL_NY = 256
PADDED = 258
STEPS = 300
SHM_ITERS = 40_000_000
DX = 1.0 / 513.0
DT = 1.0e-7
LAMBDA_DAMP = 0.15
KAPPA_MAX = 0.20  # analytical upper bound of kappa(x,y)
STRIDES = (1, 8, 16)

TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 10, 11, 12, 13

# LLVM monotonic atomic memory operations prevent Numba/LLVM from promoting
# the cross-process shared counter into a thread-local register. Each MPI rank
# writes a distinct uint64 slot, so no two ranks contend on the same word.
@intrinsic
def _atomic_load_u64(typingctx, ary, idx):
    if not isinstance(ary, types.Array) or ary.dtype != types.uint64:
        return None
    sig = types.uint64(ary, types.intp)

    def codegen(context, builder, signature, args):
        arr = context.make_array(signature.args[0])(context, builder, args[0])
        ptr = builder.gep(arr.data, [args[1]])
        # Monotonic atomic loads force a real memory operation on each loop
        # iteration; each MPI rank still owns a distinct 64-bit slot.
        return builder.load_atomic(ptr, "monotonic", 8)

    return sig, codegen


@intrinsic
def _atomic_store_u64(typingctx, ary, idx, value):
    if not isinstance(ary, types.Array) or ary.dtype != types.uint64:
        return None
    sig = types.void(ary, types.intp, types.uint64)

    def codegen(context, builder, signature, args):
        arr = context.make_array(signature.args[0])(context, builder, args[0])
        ptr = builder.gep(arr.data, [args[1]])
        builder.store_atomic(args[2], ptr, "monotonic", 8)
        return context.get_dummy_value()

    return sig, codegen


# ------------------------------- CUDA kernels --------------------------------
@cuda.jit

def initialize_fields_kernel(u, kappa, source, coord_y, coord_x):
    iy, ix = cuda.grid(2)
    if iy < PADDED and ix < PADDED:
        gy = coord_y * LOCAL_NY + (iy - 1)
        gx = coord_x * LOCAL_NX + (ix - 1)
        gyc = min(max(gy, 0), NY - 1)
        gxc = min(max(gx, 0), NX - 1)
        y = (gyc + 1.0) * DX
        x = (gxc + 1.0) * DX
        u[iy, ix] = math.exp(-80.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
        kappa[iy, ix] = 0.18 + 0.02 * math.sin(math.pi * x) * math.cos(math.pi * y)
        source[iy, ix] = 0.0


@cuda.jit

def pack_halos_kernel(u, send_up, send_down, send_left, send_right):
    idx = cuda.grid(1)
    if idx < LOCAL_NX:
        send_up[idx] = u[1, idx + 1]
        send_down[idx] = u[LOCAL_NY, idx + 1]
        send_left[idx] = u[idx + 1, 1]
        send_right[idx] = u[idx + 1, LOCAL_NX]


@cuda.jit

def unpack_halos_kernel(u, recv_up, recv_down, recv_left, recv_right,
                        has_up, has_down, has_left, has_right):
    idx = cuda.grid(1)
    if idx < LOCAL_NX:
        col = idx + 1
        row = idx + 1
        u[0, col] = recv_up[idx] if has_up else u[1, col]
        u[LOCAL_NY + 1, col] = recv_down[idx] if has_down else u[LOCAL_NY, col]
        u[row, 0] = recv_left[idx] if has_left else u[row, 1]
        u[row, LOCAL_NX + 1] = recv_right[idx] if has_right else u[row, LOCAL_NX]


@cuda.jit

def tiled_stencil_kernel(u, u_next, kappa, source, coord_y, coord_x,
                         dt, dx, damping, region):
    """16x16 block, 18x18 shared tile. region: 0=all, 1=deep, 2=border."""
    tile = cuda.shared.array(shape=(SHARED, SHARED), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tid = ty * TILE + tx

    # Cooperative load of all 324 values by 256 threads (some load two).
    for linear in range(tid, SHARED * SHARED, TILE * TILE):
        sy = linear // SHARED
        sx = linear % SHARED
        local_y = 1 + cuda.blockIdx.y * TILE + sy - 1
        local_x = 1 + cuda.blockIdx.x * TILE + sx - 1
        tile[sy, sx] = u[local_y, local_x]

    cuda.syncthreads()

    iy = 1 + cuda.blockIdx.y * TILE + ty
    ix = 1 + cuda.blockIdx.x * TILE + tx
    if iy > LOCAL_NY or ix > LOCAL_NX:
        return

    is_deep = (iy >= 2 and iy <= LOCAL_NY - 1 and
               ix >= 2 and ix <= LOCAL_NX - 1)
    if region == 1 and not is_deep:
        return
    if region == 2 and is_deep:
        return

    center = tile[ty + 1, tx + 1]
    right = tile[ty + 1, tx + 2]
    left = tile[ty + 1, tx]
    down = tile[ty + 2, tx + 1]
    up = tile[ty, tx + 1]

    kc = kappa[iy, ix]
    k_right = 0.5 * (kc + kappa[iy, ix + 1])
    k_left = 0.5 * (kc + kappa[iy, ix - 1])
    k_down = 0.5 * (kc + kappa[iy + 1, ix])
    k_up = 0.5 * (kc + kappa[iy - 1, ix])

    flux_x = (k_right * (right - center) - k_left * (center - left)) / (dx * dx)
    flux_y = (k_down * (down - center) - k_up * (center - up)) / (dx * dx)
    u_next[iy, ix] = center + dt * (flux_x + flux_y
                                     - damping * center * center * center
                                     + source[iy, ix])


# ------------------------------ CPU JIT kernel -------------------------------
@njit(nogil=True)
def host_shm_accumulate(shm_view, slot, iterations, increment):
    for _ in range(iterations):
        current = _atomic_load_u64(shm_view, slot)
        _atomic_store_u64(shm_view, slot, current + np.uint64(increment))


# ------------------------------- MPI helpers ---------------------------------
def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    )
    return out.strip().splitlines()[0].strip()


def init_hybrid_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{os.cpu_count()}"
    line_bytes = 64
    seed_input = f"{github_user.strip()}::{cpu_sig}::{gpu_uuid}"
    instance_seed = int(hashlib.sha256(seed_input.encode()).hexdigest()[:16], 16)

    cart = comm.Create_cart(dims=(2, 2), periods=(False, False), reorder=False)
    cart_rank = cart.Get_rank()
    coords = cart.Get_coords(cart_rank)
    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
    if node_comm.Get_size() != 4:
        raise RuntimeError("All four MPI ranks must be on the same shared-memory node.")
    node_rank = node_comm.Get_rank()

    # MPI shared allocations are not guaranteed to start on a 64-byte cache-line
    # boundary. Reserve 63 extra bytes and align the NumPy view *inside* the
    # returned segment. disp_unit=1 makes the queried segment size byte-based.
    shm_nbytes = 64 * np.dtype(np.uint64).itemsize
    allocation_nbytes = shm_nbytes + line_bytes - 1
    win = MPI.Win.Allocate_shared(
        allocation_nbytes if node_rank == 0 else 0,
        disp_unit=1,
        comm=node_comm,
    )
    win.Lock_all()
    shm_buf, itemsize = win.Shared_query(0)
    base_array = np.ndarray(buffer=shm_buf, dtype=np.uint8, shape=(allocation_nbytes,))
    base_address = int(base_array.ctypes.data)
    alignment_offset = (-base_address) % line_bytes
    shm_view = np.ndarray(
        buffer=shm_buf, dtype=np.uint64, shape=(64,), offset=alignment_offset
    )
    if int(shm_view.ctypes.data) % line_bytes != 0:
        raise RuntimeError("Unable to construct a 64-byte-aligned MPI shared view.")

    return {
        "comm": comm, "rank": rank, "cart": cart, "coords": coords,
        "neighbors": (up, down, left, right), "node_comm": node_comm,
        "node_rank": node_rank, "win": win, "shm_view": shm_view,
        "shm_buffer": shm_buf, "shm_alignment_offset": int(alignment_offset),
        "gpu_uuid": gpu_uuid, "cpu_signature": cpu_sig,
        "cache_line_size": line_bytes, "instance_seed": instance_seed,
    }


def allocate_gpu_state(ctx):
    u = cuda.device_array((PADDED, PADDED), dtype=np.float64)
    v = cuda.device_array((PADDED, PADDED), dtype=np.float64)
    kappa = cuda.device_array((PADDED, PADDED), dtype=np.float64)
    source = cuda.device_array((PADDED, PADDED), dtype=np.float64)
    init_stream = cuda.stream()
    initialize_fields_kernel[(17, 17), (16, 16), init_stream](
        u, kappa, source, int(ctx["coords"][0]), int(ctx["coords"][1])
    )
    init_stream.synchronize()
    return u, v, kappa, source


def make_exchange_buffers():
    send_host = [cuda.pinned_array(LOCAL_NX, dtype=np.float64) for _ in range(4)]
    recv_host = [cuda.pinned_array(LOCAL_NX, dtype=np.float64) for _ in range(4)]
    send_dev = [cuda.device_array(LOCAL_NX, dtype=np.float64) for _ in range(4)]
    recv_dev = [cuda.device_array(LOCAL_NX, dtype=np.float64) for _ in range(4)]
    return send_host, recv_host, send_dev, recv_dev


def prepare_exchange(u, send_host, send_dev, stream):
    pack_halos_kernel[1, 256, stream](u, send_dev[0], send_dev[1], send_dev[2], send_dev[3])
    for dev, host in zip(send_dev, send_host):
        dev.copy_to_host(host, stream=stream)
    stream.synchronize()


def post_exchange(ctx, send_host, recv_host):
    up, down, left, right = ctx["neighbors"]
    cart = ctx["cart"]
    reqs = [
        cart.Irecv(recv_host[0], source=up, tag=TAG_DOWN),
        cart.Irecv(recv_host[1], source=down, tag=TAG_UP),
        cart.Irecv(recv_host[2], source=left, tag=TAG_RIGHT),
        cart.Irecv(recv_host[3], source=right, tag=TAG_LEFT),
        cart.Isend(send_host[0], dest=up, tag=TAG_UP),
        cart.Isend(send_host[1], dest=down, tag=TAG_DOWN),
        cart.Isend(send_host[2], dest=left, tag=TAG_LEFT),
        cart.Isend(send_host[3], dest=right, tag=TAG_RIGHT),
    ]
    return reqs


def finish_exchange(u, recv_host, recv_dev, neighbors, stream):
    up, down, left, right = neighbors
    for dev, host in zip(recv_dev, recv_host):
        dev.copy_to_device(host, stream=stream)
    unpack_halos_kernel[1, 256, stream](
        u, recv_dev[0], recv_dev[1], recv_dev[2], recv_dev[3],
        int(up != MPI.PROC_NULL), int(down != MPI.PROC_NULL),
        int(left != MPI.PROC_NULL), int(right != MPI.PROC_NULL),
    )
    stream.synchronize()


def launch_stencil(u, v, kappa, source, coords, region, stream):
    tiled_stencil_kernel[(16, 16), (16, 16), stream](
        u, v, kappa, source, int(coords[0]), int(coords[1]),
        DT, DX, LAMBDA_DAMP, region,
    )


def global_energy(u, comm):
    host = u.copy_to_host()
    interior = host[1:257, 1:257]
    # Discrete L2 energy of the scalar field over this rank's subdomain.
    local = float(np.sum(interior * interior, dtype=np.float64) * DX * DX)
    return comm.allreduce(local, op=MPI.SUM)


def run_solver(ctx, mode, state, steps=STEPS):
    u, v, kappa, source = state
    # Reinitialize state for each independent mode.
    init_stream = cuda.stream()
    initialize_fields_kernel[(17, 17), (16, 16), init_stream](
        u, kappa, source, int(ctx["coords"][0]), int(ctx["coords"][1])
    )
    init_stream.synchronize()
    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()
    send_host, recv_host, send_dev, recv_dev = make_exchange_buffers()
    energy = {0: global_energy(u, ctx["cart"])}
    start = time.perf_counter_ns()

    for step in range(1, steps + 1):
        prepare_exchange(u, send_host, send_dev, stream_boundary)
        reqs = post_exchange(ctx, send_host, recv_host)

        if mode == "overlapped":
            # Deep interior does not depend on MPI ghost cells. Run while MPI progresses.
            launch_stencil(u, v, kappa, source, ctx["coords"], 1, stream_interior)
            MPI.Request.Waitall(reqs)
            finish_exchange(u, recv_host, recv_dev, ctx["neighbors"], stream_boundary)
            # The border uses newly unpacked halo values; output region is disjoint.
            launch_stencil(u, v, kappa, source, ctx["coords"], 2, stream_boundary)
            stream_interior.synchronize()
            stream_boundary.synchronize()
        else:
            MPI.Request.Waitall(reqs)
            finish_exchange(u, recv_host, recv_dev, ctx["neighbors"], stream_boundary)
            launch_stencil(u, v, kappa, source, ctx["coords"], 0, stream_boundary)
            stream_boundary.synchronize()

        u, v = v, u
        if step % 100 == 0:
            energy[step] = global_energy(u, ctx["cart"])

    elapsed_local_ms = (time.perf_counter_ns() - start) / 1e6
    elapsed_ms = ctx["comm"].allreduce(elapsed_local_ms, op=MPI.MAX)
    final_host = u.copy_to_host()
    return u, final_host, elapsed_ms, energy


def benchmark_shm_strides(ctx, iterations=SHM_ITERS):
    shm = ctx["shm_view"]
    node_comm = ctx["node_comm"]
    rank = ctx["node_rank"]
    seed = ctx["instance_seed"]
    results = {}

    # Compile the exact native kernel and perform a small warm-up before timing.
    shm[rank] = np.uint64(0)
    node_comm.Barrier()
    host_shm_accumulate(shm, rank, 2, np.uint64(1 + (seed + rank) % 7))
    node_comm.Barrier()

    for stride in STRIDES:
        slot = rank * stride
        inc = np.uint64(1 + ((seed + rank + stride) % 7))
        if rank == 0:
            shm[:64] = np.uint64(0)
        ctx["win"].Sync()
        node_comm.Barrier()
        start = time.perf_counter_ns()
        host_shm_accumulate(shm, slot, iterations, inc)
        local_ns = time.perf_counter_ns() - start
        ctx["win"].Sync()
        node_comm.Barrier()
        elapsed_ns = node_comm.allreduce(local_ns, op=MPI.MAX)
        expected = np.uint64(iterations) * inc
        parity = int(shm[slot]) == int(expected)
        all_parity = node_comm.allreduce(int(parity), op=MPI.MIN)
        results[stride] = {
            "elapsed_ns": int(elapsed_ns),
            "ns_per_update": float(elapsed_ns / iterations),
            "final_counter": int(shm[slot]),
            "expected_counter": int(expected),
            "parity_passed": bool(all_parity),
            "increment": int(inc),
        }

    # LLVM audit on the compiled Numba function.
    sigs = host_shm_accumulate.signatures
    llvm = host_shm_accumulate.inspect_llvm(sigs[0]) if sigs else ""
    atomic_loads = len(re.findall(r"load\s+atomic\s+i64", llvm))
    atomic_stores = len(re.findall(r"store\s+atomic\s+i64", llvm))
    llvm_excerpt = "\n".join(line for line in llvm.splitlines()
                              if "atomic" in line and ("load" in line or "store" in line))[:1500]
    llvm_audit = {
        "atomic_i64_load_count": atomic_loads,
        "atomic_i64_store_count": atomic_stores,
        "per_iteration_memory_accesses_verified": atomic_loads > 0 and atomic_stores > 0,
        "llvm_ir_excerpt": llvm_excerpt,
    }
    base = results[8]["elapsed_ns"]
    for stride in STRIDES:
        results[stride]["ratio_vs_stride_8"] = float(results[stride]["elapsed_ns"] / base)
    results["false_sharing_slowdown_stride1_vs_stride8"] = float(
        results[1]["elapsed_ns"] / results[8]["elapsed_ns"]
    )
    return results, llvm_audit


def audit_ptx():
    signatures = tiled_stencil_kernel.signatures
    if not signatures:
        raise RuntimeError("CUDA stencil kernel was not compiled; cannot audit PTX.")
    asm = tiled_stencil_kernel.inspect_asm(signatures[0])
    counts = {
        "st.shared.f64": asm.count("st.shared.f64"),
        "ld.shared.f64": asm.count("ld.shared.f64"),
        "bar.sync": asm.count("bar.sync"),
    }
    counts["passed"] = all(counts[k] > 0 for k in ("st.shared.f64", "ld.shared.f64", "bar.sync"))
    counts["ptx_excerpt"] = "\n".join(
        line for line in asm.splitlines()
        if any(token in line for token in ("st.shared", "ld.shared", "bar.sync"))
    )[:2000]
    return counts


def main():
    github_user = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    if not github_user or github_user == "REPLACE_WITH_YOUR_GITHUB_USERNAME":
        raise RuntimeError("Pass your real GitHub username as the first argument.")

    ctx = init_hybrid_context(github_user)
    state = allocate_gpu_state(ctx)

    # Compile and execute a warm-up full stencil so PTX is available for inspection.
    warm_stream = cuda.stream()
    launch_stencil(*state, ctx["coords"], 0, warm_stream)
    warm_stream.synchronize()
    ptx_audit = audit_ptx()

    serialized_u, serialized_host, serialized_ms, serialized_energy = run_solver(
        ctx, "serialized", state
    )
    overlapped_u, overlapped_host, overlapped_ms, overlapped_energy = run_solver(
        ctx, "overlapped", state
    )
    local_diff = float(np.max(np.abs(serialized_host[1:257, 1:257]
                                     - overlapped_host[1:257, 1:257])))
    max_diff = ctx["cart"].allreduce(local_diff, op=MPI.MAX)
    energy_monotonic = all(
        vals[step] >= vals[next_step] - 1e-14
        for vals in (serialized_energy, overlapped_energy)
        for step, next_step in ((0, 100), (100, 200), (200, 300))
    )

    shm_results, llvm_audit = benchmark_shm_strides(ctx)
    kappa_max = float(KAPPA_MAX)
    dt_max = (DX * DX) / (4.0 * kappa_max)
    cfl_stable = DT <= dt_max
    local_shm_parity = all(shm_results[s]["parity_passed"] for s in STRIDES)
    shm_parity = ctx["comm"].allreduce(int(local_shm_parity), op=MPI.MIN)
    verification = bool(
        max_diff < 1e-12 and energy_monotonic and ptx_audit["passed"]
        and llvm_audit["per_iteration_memory_accesses_verified"]
        and shm_parity and cfl_stable
    )

    # Gather only scalar telemetry/metadata, never distributed field arrays.
    rank_record = {
        "rank": ctx["rank"], "cart_coords": list(ctx["coords"]),
        "neighbors": [int(x) for x in ctx["neighbors"]],
    }
    rank_records = ctx["comm"].gather(rank_record, root=0)

    if ctx["rank"] == 0:
        manifest = {
            "github_username": github_user,
            "instance_seed": ctx["instance_seed"],
            "cpu_signature": ctx["cpu_signature"],
            "logical_cpus": os.cpu_count(),
            "cpu_cache_line_bytes": ctx["cache_line_size"],
            "shared_window_alignment_offset_bytes": ctx["shm_alignment_offset"],
            "shared_window_is_64_byte_aligned": bool(
                int(ctx["shm_view"].ctypes.data) % ctx["cache_line_size"] == 0
            ),
            "gpu_name": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
            ).strip().splitlines()[0],
            "gpu_uuid": ctx["gpu_uuid"],
            "gpu_compute_capability": ".".join(map(str, cuda.get_current_device().compute_capability)),
            "mpi_ranks": 4,
            "cartesian_dimensions": [2, 2],
            "rank_topology": rank_records,
            "global_grid": [NY, NX],
            "local_interior_per_rank": [LOCAL_NY, LOCAL_NX],
            "padded_local_shape": [PADDED, PADDED],
            "stencil_steps": STEPS,
            "dt": DT,
            "dx": DX,
            "kappa_max": kappa_max,
            "dt_max": dt_max,
            "cfl_stable": bool(cfl_stable),
            "gpu_serialized_ms": float(serialized_ms),
            "gpu_overlapped_ms": float(overlapped_ms),
            "gpu_overlap_speedup": float(serialized_ms / overlapped_ms),
            "max_diff_serialized_vs_overlapped": float(max_diff),
            "serialized_global_energy": {str(k): float(v) for k, v in serialized_energy.items()},
            "overlapped_global_energy": {str(k): float(v) for k, v in overlapped_energy.items()},
            "global_energy_monotonic": bool(energy_monotonic),
            "ptx_audit": ptx_audit,
            "shared_memory_stride_benchmark": shm_results,
            "cpu_llvm_audit": llvm_audit,
            "shared_memory_parity_passed": bool(shm_parity),
            "verification_passed": verification,
        }
        Path("hybrid_lab_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print("\n===== HYBRID MPI-CUDA RESULTS =====")
        print(f"GPU serialized: {serialized_ms:.3f} ms")
        print(f"GPU overlapped: {overlapped_ms:.3f} ms")
        print(f"Overlap speedup: {serialized_ms / overlapped_ms:.3f}x")
        print(f"Max parity difference: {max_diff:.3e}")
        print("Serialized energy:", serialized_energy)
        print("Overlapped energy:", overlapped_energy)
        print("PTX audit:", {k: v for k, v in ptx_audit.items() if k != "ptx_excerpt"})
        print("\n===== MPI SHARED-MEMORY STRIDE BENCHMARK =====")
        for stride in STRIDES:
            print(f"stride={stride:2d}: {shm_results[stride]}")
        print("False-sharing ratio (stride 1 / stride 8):",
              shm_results["false_sharing_slowdown_stride1_vs_stride8"])
        print("CPU LLVM audit:", {k: v for k, v in llvm_audit.items() if k != "llvm_ir_excerpt"})
        print("\nVerification passed:", verification)
        print("hybrid_lab_manifest.json generated.")

    ctx["win"].Unlock_all()
    ctx["win"].Free()
    ctx["node_comm"].Free()
    ctx["cart"].Free()
    MPI.Finalize()
    if not verification:
        # All ranks agree because all checks are based on global reductions.
        raise SystemExit(2)


if __name__ == "__main__":
    main()

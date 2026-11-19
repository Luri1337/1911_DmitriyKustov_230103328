
import os
import sys
import json
import math
import hashlib
import subprocess

import numpy as np
from mpi4py import MPI
from numba import cuda, float64

# ============================================================
# CONSTANTS
# ============================================================

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2

GLOBAL_NX = 512
GLOBAL_NY = 512

LOCAL_NX = 256
LOCAL_NY = 256

STEPS = 400
LAMBDA_DAMP = 0.15

DX = 1.0 / 513.0
DY = 1.0 / 513.0

# ============================================================
# GPU UUID AND STUDENT SEED
# ============================================================

def get_gpu_uuid():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader"
        ],
        encoding="utf-8"
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user, gpu_uuid):
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")
    ).hexdigest()

    return int(digest[:16], 16)


# ============================================================
# TASK 1: DISTRIBUTED CONTEXT
# ============================================================

def init_distributed_context(github_user):

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 4:
        raise RuntimeError(
            f"Expected exactly 4 MPI ranks, got {size}"
        )

    cuda.select_device(0)

    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)

    instance_seed = derive_student_seed(
        github_user,
        gpu_uuid
    )

    cart_comm = comm.Create_cart(
        dims=[2, 2],
        periods=[False, False],
        reorder=False
    )

    coords = cart_comm.Get_coords(rank)

    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    return {
        "comm": comm,
        "cart_comm": cart_comm,
        "rank": rank,
        "size": size,
        "coords": coords,
        "neighbors": {
            "up": nbr_up,
            "down": nbr_down,
            "left": nbr_left,
            "right": nbr_right
        },
        "gpu_uuid": gpu_uuid,
        "instance_seed": instance_seed
    }


# ============================================================
# TASK 2: SUBDOMAIN INITIALIZATION
# ============================================================

def initialize_subdomains(context):

    rank = context["rank"]
    row_rank, col_rank = context["coords"]

    seed = context["instance_seed"]

    rng = np.random.default_rng(seed + rank)

    # Local coordinates including ghost cells.
    global_rows = (
        row_rank * LOCAL_NY
        + np.arange(258)
    )

    global_cols = (
        col_rank * LOCAL_NX
        + np.arange(258)
    )

    y = global_rows[:, None] * DY
    x = global_cols[None, :] * DX

    # Spatially heterogeneous diffusivity.
    kappa = (
        0.16
        + 0.04 * np.sin(2 * np.pi * x)
        * np.cos(2 * np.pi * y)
        + 0.02 * np.sin(4 * np.pi * x * y)
    )

    kappa = np.clip(kappa, 0.08, 0.24)

    # Seed-dependent Gaussian thermal pulses.
    u = np.zeros((258, 258), dtype=np.float64)

    pulse_count = 4

    for _ in range(pulse_count):

        center_x = rng.uniform(0.15, 0.85)
        center_y = rng.uniform(0.15, 0.85)

        amplitude = rng.uniform(0.5, 1.0)
        sigma = rng.uniform(0.04, 0.09)

        pulse = amplitude * np.exp(
            -(
                (x - center_x) ** 2
                + (y - center_y) ** 2
            ) / (2 * sigma ** 2)
        )

        u += pulse

    # Localized source field.
    source = np.zeros((258, 258), dtype=np.float64)

    source_x = 0.5
    source_y = 0.5
    source_sigma = 0.06

    source = np.exp(
        -(
            (x - source_x) ** 2
            + (y - source_y) ** 2
        ) / (2 * source_sigma ** 2)
    )

    # Physical Dirichlet boundaries.
    neighbors = context["neighbors"]

    if neighbors["up"] == MPI.PROC_NULL:
        u[0, :] = 0.0

    if neighbors["down"] == MPI.PROC_NULL:
        u[-1, :] = 0.0

    if neighbors["left"] == MPI.PROC_NULL:
        u[:, 0] = 0.0

    if neighbors["right"] == MPI.PROC_NULL:
        u[:, -1] = 0.0

    return {
        "u": u,
        "kappa": kappa,
        "source": source,
        "x": x,
        "y": y
    }


# ============================================================
# INITIALIZATION TEST
# ============================================================

def initialization_test(github_user):

    context = init_distributed_context(github_user)

    fields = initialize_subdomains(context)

    comm = context["cart_comm"]

    print(
        f"Rank {context['rank']} | "
        f"Coordinates: {context['coords']} | "
        f"Neighbors: {context['neighbors']} | "
        f"GPU: {context['gpu_uuid']}"
    )

    print(
        f"Rank {context['rank']} | "
        f"u shape: {fields['u'].shape} | "
        f"kappa range: "
        f"[{fields['kappa'].min():.6f}, "
        f"{fields['kappa'].max():.6f}] | "
        f"Seed: {context['instance_seed']}"
    )

    comm.Barrier()

    if context["rank"] == 0:
        print("\nTask 1 and Task 2 initialization completed.")


if __name__ == "__main__":

    github_user = sys.argv[1]

    initialization_test(github_user)

# ============================================================
# TASK 3: CUDA HALO PACK / UNPACK
# ============================================================

@cuda.jit
def pack_halos_kernel(u, buffer, n, direction):
    idx = cuda.grid(1)

    if idx < n:
        if direction == 0:       # Top row
            buffer[idx] = u[1, idx + 1]

        elif direction == 1:     # Bottom row
            buffer[idx] = u[n, idx + 1]

        elif direction == 2:     # Left column
            buffer[idx] = u[idx + 1, 1]

        elif direction == 3:     # Right column
            buffer[idx] = u[idx + 1, n]


@cuda.jit
def unpack_halos_kernel(u, buffer, n, direction):
    idx = cuda.grid(1)

    if idx < n:
        if direction == 0:       # Top ghost row
            u[0, idx + 1] = buffer[idx]

        elif direction == 1:     # Bottom ghost row
            u[n + 1, idx + 1] = buffer[idx]

        elif direction == 2:     # Left ghost column
            u[idx + 1, 0] = buffer[idx]

        elif direction == 3:     # Right ghost column
            u[idx + 1, n + 1] = buffer[idx]


# ============================================================
# PINNED HOST BUFFERS AND DEVICE BUFFERS
# ============================================================

def create_halo_buffers(n=256):

    send_host = [
        cuda.pinned_array(n, dtype=np.float64)
        for _ in range(4)
    ]

    recv_host = [
        cuda.pinned_array(n, dtype=np.float64)
        for _ in range(4)
    ]

    send_device = [
        cuda.device_array(n, dtype=np.float64)
        for _ in range(4)
    ]

    recv_device = [
        cuda.device_array(n, dtype=np.float64)
        for _ in range(4)
    ]

    return {
        "send_host": send_host,
        "recv_host": recv_host,
        "send_device": send_device,
        "recv_device": recv_device
    }


# ============================================================
# ASYNCHRONOUS HALO EXCHANGE
# ============================================================

def exchange_halos_async(
    u,
    context,
    buffers,
    stream_boundary
):

    n = LOCAL_NX

    cart_comm = context["cart_comm"]
    neighbors = context["neighbors"]

    send_host = buffers["send_host"]
    recv_host = buffers["recv_host"]

    send_device = buffers["send_device"]
    recv_device = buffers["recv_device"]

    # Pack four owned borders on GPU.
    threads = 256
    blocks = (n + threads - 1) // threads

    for direction in range(4):

        pack_halos_kernel[blocks, threads, stream_boundary](
            u,
            send_device[direction],
            n,
            direction
        )

        send_device[direction].copy_to_host(
            send_host[direction],
            stream=stream_boundary
        )

    # Ensure D2H transfers are complete before MPI reads buffers.
    stream_boundary.synchronize()

    # Directions:
    # 0 = up
    # 1 = down
    # 2 = left
    # 3 = right

    neighbor_list = [
        neighbors["up"],
        neighbors["down"],
        neighbors["left"],
        neighbors["right"]
    ]

    # Opposite directions for incoming halos.
    opposite = [1, 0, 3, 2]

    requests = []

    # Post all receives first.
    for direction in range(4):

        neighbor = neighbor_list[direction]

        if neighbor != MPI.PROC_NULL:

            request = cart_comm.Irecv(
                recv_host[direction],
                source=neighbor,
                tag=opposite[direction]
            )

            requests.append(request)

    # Post all sends.
    for direction in range(4):

        neighbor = neighbor_list[direction]

        if neighbor != MPI.PROC_NULL:

            request = cart_comm.Isend(
                send_host[direction],
                dest=neighbor,
                tag=direction
            )

            requests.append(request)

    # Wait for all communication.
    MPI.Request.Waitall(requests)

    # Transfer received halos to GPU.
    for direction in range(4):

        if neighbor_list[direction] != MPI.PROC_NULL:

            recv_device[direction].copy_to_device(
                recv_host[direction],
                stream=stream_boundary
            )

            unpack_halos_kernel[blocks, threads, stream_boundary](
                u,
                recv_device[direction],
                n,
                direction
            )

    stream_boundary.synchronize()

    return u


# ============================================================
# TASK 3: HALO EXCHANGE TEST
# ============================================================

def halo_exchange_test(github_user):

    context = init_distributed_context(github_user)

    fields = initialize_subdomains(context)

    # Upload local field to GPU.
    u_device = cuda.to_device(fields["u"])

    stream_boundary = cuda.stream()

    buffers = create_halo_buffers(LOCAL_NX)

    # Perform one complete halo exchange.
    exchange_halos_async(
        u_device,
        context,
        buffers,
        stream_boundary
    )

    result = u_device.copy_to_host()

    cart_comm = context["cart_comm"]
    cart_comm.Barrier()

    rank = context["rank"]

    print(
        f"Rank {rank}: "
        f"Halo exchange completed | "
        f"Shape: {result.shape} | "
        f"Top ghost mean: {result[0, 1:-1].mean():.8f} | "
        f"Bottom ghost mean: {result[-1, 1:-1].mean():.8f}",
        flush=True
    )

    cart_comm.Barrier()

    if rank == 0:
        print("Task 3 halo exchange test completed.", flush=True)

# ============================================================
# TASK 4: SHARED MEMORY TILED STENCIL
# ============================================================

@cuda.jit
def tiled_stencil_kernel(
    u,
    kappa,
    source,
    u_next,
    dt,
    dx,
    dy,
    r_start,
    r_end,
    c_start,
    c_end
):

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    row = r_start + cuda.blockIdx.y * 16 + ty
    col = c_start + cuda.blockIdx.x * 16 + tx

    # Shared memory tile: 16x16 + 1-cell halo apron.
    smem = cuda.shared.array(
        shape=(18, 18),
        dtype=float64
    )

    # Load center cells.
    if row <= 257 and col <= 257:
        smem[ty + 1, tx + 1] = u[row, col]
    else:
        smem[ty + 1, tx + 1] = 0.0

    # Load halo apron.
    if ty == 0:
        if row - 1 >= 0 and col <= 257:
            smem[0, tx + 1] = u[row - 1, col]
        else:
            smem[0, tx + 1] = 0.0

    if ty == 15:
        if row + 1 <= 257 and col <= 257:
            smem[17, tx + 1] = u[row + 1, col]
        else:
            smem[17, tx + 1] = 0.0

    if tx == 0:
        if col - 1 >= 0 and row <= 257:
            smem[ty + 1, 0] = u[row, col - 1]
        else:
            smem[ty + 1, 0] = 0.0

    if tx == 15:
        if col + 1 <= 257 and row <= 257:
            smem[ty + 1, 17] = u[row, col + 1]
        else:
            smem[ty + 1, 17] = 0.0

    # Synchronize all 256 threads.
    cuda.syncthreads()

    # Update only cells inside the requested bounding box.
    if (
        row >= r_start
        and row < r_end
        and col >= c_start
        and col < c_end
        and row >= 1
        and row <= 256
        and col >= 1
        and col <= 256
    ):

        center = smem[ty + 1, tx + 1]

        up = smem[ty, tx + 1]
        down = smem[ty + 2, tx + 1]

        left = smem[ty + 1, tx]
        right = smem[ty + 1, tx + 2]

        k_center = kappa[row, col]

        k_up = 0.5 * (k_center + kappa[row - 1, col])
        k_down = 0.5 * (k_center + kappa[row + 1, col])

        k_left = 0.5 * (k_center + kappa[row, col - 1])
        k_right = 0.5 * (k_center + kappa[row, col + 1])

        diffusion_x = (
            k_down * (down - center)
            - k_up * (center - up)
        ) / (dx * dx)

        diffusion_y = (
            k_right * (right - center)
            - k_left * (center - left)
        ) / (dy * dy)

        damping = 0.15 * center * center * center

        value = center + dt * (
            diffusion_x
            + diffusion_y
            - damping
            + source[row, col]
        )

        u_next[row, col] = value


# ============================================================
# STENCIL LAUNCH HELPER
# ============================================================

def launch_stencil(
    u,
    kappa,
    source,
    u_next,
    dt,
    stream,
    r_start,
    r_end,
    c_start,
    c_end
):

    rows = r_end - r_start
    cols = c_end - c_start

    blocks = (
        math.ceil(cols / 16),
        math.ceil(rows / 16)
    )

    threads = (16, 16)

    tiled_stencil_kernel[blocks, threads, stream](
        u,
        kappa,
        source,
        u_next,
        dt,
        DX,
        DY,
        r_start,
        r_end,
        c_start,
        c_end
    )

# ============================================================
# TASK 4: DUAL-STREAM COMPUTE / COMMUNICATION OVERLAP
# ============================================================

def begin_halo_exchange(
    u,
    context,
    buffers,
    stream_boundary
):

    n = LOCAL_NX
    cart_comm = context["cart_comm"]
    neighbors = context["neighbors"]

    send_host = buffers["send_host"]
    recv_host = buffers["recv_host"]

    send_device = buffers["send_device"]

    threads = 256
    blocks = (n + threads - 1) // threads

    # Stage A: pack and asynchronously transfer borders D2H.
    for direction in range(4):

        pack_halos_kernel[blocks, threads, stream_boundary](
            u,
            send_device[direction],
            n,
            direction
        )

        send_device[direction].copy_to_host(
            send_host[direction],
            stream=stream_boundary
        )

    stream_boundary.synchronize()

    neighbor_list = [
        neighbors["up"],
        neighbors["down"],
        neighbors["left"],
        neighbors["right"]
    ]

    opposite = [1, 0, 3, 2]

    requests = []

    # Stage B: post receives.
    for direction in range(4):

        neighbor = neighbor_list[direction]

        if neighbor != MPI.PROC_NULL:

            requests.append(
                cart_comm.Irecv(
                    recv_host[direction],
                    source=neighbor,
                    tag=opposite[direction]
                )
            )

    # Stage B: post sends.
    for direction in range(4):

        neighbor = neighbor_list[direction]

        if neighbor != MPI.PROC_NULL:

            requests.append(
                cart_comm.Isend(
                    send_host[direction],
                    dest=neighbor,
                    tag=direction
                )
            )

    return requests


def finish_halo_exchange(
    u,
    context,
    buffers,
    requests,
    stream_boundary
):

    neighbors = context["neighbors"]

    recv_host = buffers["recv_host"]
    recv_device = buffers["recv_device"]

    neighbor_list = [
        neighbors["up"],
        neighbors["down"],
        neighbors["left"],
        neighbors["right"]
    ]

    # Stage C: wait for MPI transfers.
    MPI.Request.Waitall(requests)

    threads = 256
    blocks = (LOCAL_NX + threads - 1) // threads

    # Transfer received halos H2D and unpack.
    for direction in range(4):

        if neighbor_list[direction] != MPI.PROC_NULL:

            recv_device[direction].copy_to_device(
                recv_host[direction],
                stream=stream_boundary
            )

            unpack_halos_kernel[blocks, threads, stream_boundary](
                u,
                recv_device[direction],
                LOCAL_NX,
                direction
            )


def step_overlapped(
    u,
    u_next,
    kappa,
    source,
    context,
    buffers,
    dt,
    stream_interior,
    stream_boundary
):

    # Start asynchronous halo communication.
    requests = begin_halo_exchange(
        u,
        context,
        buffers,
        stream_boundary
    )

    # Concurrently compute deep interior.
    launch_stencil(
        u,
        kappa,
        source,
        u_next,
        dt,
        stream_interior,
        2, 256,
        2, 256
    )

    # Wait for halo communication and unpack.
    finish_halo_exchange(
        u,
        context,
        buffers,
        requests,
        stream_boundary
    )

    # Wait for unpack operations before boundary computation.
    stream_boundary.synchronize()

    # Compute top perimeter.
    launch_stencil(
        u, kappa, source, u_next,
        dt, stream_boundary,
        1, 2, 1, 257
    )

    # Compute bottom perimeter.
    launch_stencil(
        u, kappa, source, u_next,
        dt, stream_boundary,
        256, 257, 1, 257
    )

    # Compute left perimeter.
    launch_stencil(
        u, kappa, source, u_next,
        dt, stream_boundary,
        2, 256, 1, 2
    )

    # Compute right perimeter.
    launch_stencil(
        u, kappa, source, u_next,
        dt, stream_boundary,
        2, 256, 256, 257
    )

    # Both streams must finish before swapping buffers.
    stream_interior.synchronize()
    stream_boundary.synchronize()

    return u_next


def step_serialized(
    u,
    u_next,
    kappa,
    source,
    context,
    buffers,
    dt,
    stream_boundary
):

    requests = begin_halo_exchange(
        u,
        context,
        buffers,
        stream_boundary
    )

    finish_halo_exchange(
        u,
        context,
        buffers,
        requests,
        stream_boundary
    )

    stream_boundary.synchronize()

    # Update the complete owned interior.
    launch_stencil(
        u,
        kappa,
        source,
        u_next,
        dt,
        stream_boundary,
        1, 257,
        1, 257
    )

    stream_boundary.synchronize()

    return u_next

def distributed_step_test(github_user):

    context = init_distributed_context(github_user)
    fields = initialize_subdomains(context)

    u = cuda.to_device(fields["u"])
    kappa = cuda.to_device(fields["kappa"])
    source = cuda.to_device(fields["source"])

    u_next = cuda.device_array(
        (258, 258),
        dtype=np.float64
    )

    buffers = create_halo_buffers(LOCAL_NX)

    stream_interior = cuda.stream()
    stream_boundary = cuda.stream()

    dt = 1e-7

    result = step_overlapped(
        u,
        u_next,
        kappa,
        source,
        context,
        buffers,
        dt,
        stream_interior,
        stream_boundary
    )

    host_result = result.copy_to_host()

    context["cart_comm"].Barrier()

    print(
        f"Rank {context['rank']} | "
        f"Step completed | "
        f"Finite: {np.all(np.isfinite(host_result))} | "
        f"Max: {host_result.max():.8f}",
        flush=True
    )


if __name__ == "__main__":

    if len(sys.argv) > 2 and sys.argv[2] == "step_test":
        distributed_step_test(sys.argv[1])
    else:
        initialization_test(sys.argv[1])

def parity_test(github_user):

    context = init_distributed_context(github_user)
    fields = initialize_subdomains(context)

    # Identical initial states.
    u_serial = cuda.to_device(fields["u"])
    u_overlap = cuda.to_device(fields["u"])

    kappa = cuda.to_device(fields["kappa"])
    source = cuda.to_device(fields["source"])

    next_serial = cuda.device_array(
        (258, 258), dtype=np.float64
    )

    next_overlap = cuda.device_array(
        (258, 258), dtype=np.float64
    )

    next_serial.copy_to_device(u_serial)
    next_overlap.copy_to_device(u_overlap)

    buffers_serial = create_halo_buffers()
    buffers_overlap = create_halo_buffers()

    stream_serial = cuda.stream()
    stream_interior = cuda.stream()
    stream_boundary = cuda.stream()

    dt = 1e-7

    # Serialized execution.
    step_serialized(
        u_serial,
        next_serial,
        kappa,
        source,
        context,
        buffers_serial,
        dt,
        stream_serial
    )

    # Overlapped execution.
    step_overlapped(
        u_overlap,
        next_overlap,
        kappa,
        source,
        context,
        buffers_overlap,
        dt,
        stream_interior,
        stream_boundary
    )

    serial_result = next_serial.copy_to_host()
    overlap_result = next_overlap.copy_to_host()

    local_diff = np.max(
        np.abs(serial_result - overlap_result)
    )

    global_diff = context["cart_comm"].allreduce(
        local_diff,
        op=MPI.MAX
    )

    print(
        f"Rank {context['rank']} | "
        f"Local difference: {local_diff:.15e}",
        flush=True
    )

    context["cart_comm"].Barrier()

    if context["rank"] == 0:
        print(
            f"\nGlobal maximum difference: {global_diff:.15e}",
            flush=True
        )
        print(
            "Parity passed:",
            global_diff < 1e-12,
            flush=True
        )

# ============================================================
# TASK 5: DISTRIBUTED SOLVER
# ============================================================

def calculate_global_metrics(
    u,
    previous_state,
    cart_comm
):
    host = u.copy_to_host()

    local_energy = (
        np.sum(host[1:257, 1:257])
        * DX * DY
    )

    if previous_state is None:
        local_residual = 0.0
    else:
        delta = (
            host[1:257, 1:257]
            - previous_state
        )

        local_residual = np.sum(delta * delta) * DX * DY

    global_energy = cart_comm.allreduce(
        local_energy,
        op=MPI.SUM
    )

    global_residual = cart_comm.allreduce(
        local_residual,
        op=MPI.SUM
    )

    return (
        float(global_energy),
        float(math.sqrt(global_residual))
    )


def run_mode(
    initial_fields,
    context,
    mode,
    dt=1e-7,
    steps=400
):

    cart_comm = context["cart_comm"]

    u = cuda.to_device(initial_fields["u"])

    u_next = cuda.to_device(initial_fields["u"])

    kappa = cuda.to_device(initial_fields["kappa"])
    source = cuda.to_device(initial_fields["source"])

    buffers = create_halo_buffers(LOCAL_NX)

    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()

    # CUDA event timing.
    start_boundary = cuda.event(timing=True)
    stop_boundary = cuda.event(timing=True)

    start_interior = cuda.event(timing=True)
    stop_interior = cuda.event(timing=True)

    cart_comm.Barrier()

    wall_start = MPI.Wtime()

    start_boundary.record(stream_boundary)
    start_interior.record(stream_interior)

    energy_trace = []
    residual_trace = []

    # Step 0 diagnostics.
    energy, residual = calculate_global_metrics(
        u,
        None,
        cart_comm
    )

    energy_trace.append((0, energy))
    residual_trace.append((0, residual))

    if context["rank"] == 0:
        print(
            f"{mode} | Step 0 | "
            f"Energy={energy:.12e} | "
            f"L2={residual:.12e}",
            flush=True
        )

    for step in range(1, steps + 1):

        # Save the previous timestep only when the next
        # iteration is a checkpoint.
        previous_state = None

        if step % 100 == 0:
            previous_state = u.copy_to_host()[1:257, 1:257].copy()

        if mode == "serialized":

            step_serialized(
                u,
                u_next,
                kappa,
                source,
                context,
                buffers,
                dt,
                stream_boundary
            )

        elif mode == "overlapped":

            step_overlapped(
                u,
                u_next,
                kappa,
                source,
                context,
                buffers,
                dt,
                stream_interior,
                stream_boundary
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Swap current and next fields.
        u, u_next = u_next, u

        if step % 100 == 0:

            energy, residual = calculate_global_metrics(
                u,
                previous_state,
                cart_comm
            )

            energy_trace.append((step, energy))
            residual_trace.append((step, residual))

            if context["rank"] == 0:
                print(
                    f"{mode} | Step {step} | "
                    f"Energy={energy:.12e} | "
                    f"L2={residual:.12e}",
                    flush=True
                )

    stream_boundary.synchronize()
    stream_interior.synchronize()

    stop_boundary.record(stream_boundary)
    stop_interior.record(stream_interior)

    stop_boundary.synchronize()
    stop_interior.synchronize()

    local_wall_time = MPI.Wtime() - wall_start

    global_wall_time = cart_comm.allreduce(
        local_wall_time,
        op=MPI.MAX
    )

    boundary_ms = start_boundary.elapsed_time(stop_boundary)
    interior_ms = start_interior.elapsed_time(stop_interior)

    local_cuda_ms = max(boundary_ms, interior_ms)

    global_cuda_ms = cart_comm.allreduce(
        local_cuda_ms,
        op=MPI.MAX
    )

    final_state = u.copy_to_host()

    return {
        "final_state": final_state,
        "energy_trace": energy_trace,
        "residual_trace": residual_trace,
        "wall_time_ms": global_wall_time * 1000.0,
        "cuda_event_ms": global_cuda_ms
    }


# ============================================================
# PTX ASSEMBLY AUDIT
# ============================================================

def audit_stencil_ptx():

    dummy = np.zeros((258, 258), dtype=np.float64)

    d_u = cuda.to_device(dummy)
    d_k = cuda.to_device(dummy)
    d_s = cuda.to_device(dummy)
    d_out = cuda.to_device(dummy)

    stream = cuda.stream()

    launch_stencil(
        d_u, d_k, d_s, d_out,
        1e-7, stream,
        2, 256, 2, 256
    )

    stream.synchronize()

    ptx_result = tiled_stencil_kernel.inspect_asm()

    # Support both string and dictionary results.
    if isinstance(ptx_result, dict):
        ptx = "\n".join(
            value for value in ptx_result.values()
            if isinstance(value, str)
        )
    else:
        ptx = ptx_result

    import re

    counts = {
        "st.shared.f64": len(re.findall(r"\bst\.shared\.f64\b", ptx)),
        "ld.shared.f64": len(re.findall(r"\bld\.shared\.f64\b", ptx)),
        "bar.sync": len(re.findall(r"\bbar\.sync\b", ptx))
    }

    return counts, ptx


# ============================================================
# TASK 5: COMPLETE SOLVER
# ============================================================

def run_solver(github_user):

    context = init_distributed_context(github_user)

    rank = context["rank"]
    cart_comm = context["cart_comm"]

    fields = initialize_subdomains(context)

    dt = 1e-7
    steps = 400

    local_kappa_max = float(fields["kappa"].max())

    global_kappa_max = cart_comm.allreduce(
        local_kappa_max,
        op=MPI.MAX
    )

    dt_max = DX**2 / (4.0 * global_kappa_max)

    if rank == 0:
        print("\n===== CFL STABILITY =====")
        print(f"kappa_max: {global_kappa_max:.12f}")
        print(f"dt:        {dt:.12e}")
        print(f"dt_max:    {dt_max:.12e}")
        print(f"Stable:    {dt < dt_max}", flush=True)

    # Run serialized mode.
    serialized = run_mode(
        fields,
        context,
        "serialized",
        dt,
        steps
    )

    cart_comm.Barrier()

    # Run overlapped mode.
    overlapped = run_mode(
        fields,
        context,
        "overlapped",
        dt,
        steps
    )

    # Compare final states.
    local_difference = float(
        np.max(
            np.abs(
                serialized["final_state"]
                - overlapped["final_state"]
            )
        )
    )

    parity_max_diff = cart_comm.allreduce(
        local_difference,
        op=MPI.MAX
    )

    # PTX audit on each process.
    ptx_counts, ptx = audit_stencil_ptx()

    audit_passed = (
        ptx_counts["st.shared.f64"] > 0
        and ptx_counts["ld.shared.f64"] > 0
        and ptx_counts["bar.sync"] > 0
    )

    audit_results = cart_comm.allgather(
        (ptx_counts, audit_passed)
    )

    if rank == 0:

        print("\n===== BENCHMARK =====")

        print(
            f"Serialized wall time: "
            f"{serialized['wall_time_ms']:.3f} ms"
        )

        print(
            f"Overlapped wall time: "
            f"{overlapped['wall_time_ms']:.3f} ms"
        )

        print(
            f"Serialized CUDA event time: "
            f"{serialized['cuda_event_ms']:.3f} ms"
        )

        print(
            f"Overlapped CUDA event time: "
            f"{overlapped['cuda_event_ms']:.3f} ms"
        )

        speedup = (
            serialized["wall_time_ms"]
            / overlapped["wall_time_ms"]
        )

        print(f"Speedup: {speedup:.6f}x")

        print("\n===== PARITY =====")

        print(
            f"Maximum absolute difference: "
            f"{parity_max_diff:.15e}"
        )

        print(
            "Parity passed:",
            parity_max_diff < 1e-12
        )

        print("\n===== PTX AUDIT =====")

        print(json.dumps(ptx_counts, indent=2))
        print("PTX audit passed:", audit_passed)

        print("\n===== FINAL ENERGY TRACE =====")

        for step, energy in overlapped["energy_trace"]:
            print(f"Step {step}: Energy={energy:.12e}")

        print("\n===== MANIFEST GENERATION =====")

        manifest = {
            "github_username": github_user,
            "gpu_uuid": context["gpu_uuid"],
            "instance_seed": context["instance_seed"],
            "mpi_ranks": context["size"],
            "cartesian_grid": [2, 2],
            "local_shape": [258, 258],
            "steps": steps,
            "dt": dt,
            "kappa_max": global_kappa_max,
            "dt_max": dt_max,
            "cfl_stable": bool(dt < dt_max),
            "serialized_ms": serialized["wall_time_ms"],
            "overlapped_ms": overlapped["wall_time_ms"],
            "serialized_cuda_event_ms": serialized["cuda_event_ms"],
            "overlapped_cuda_event_ms": overlapped["cuda_event_ms"],
            "speedup": speedup,
            "parity_max_diff": parity_max_diff,
            "parity_passed": bool(parity_max_diff < 1e-12),
            "ptx_counts": ptx_counts,
            "ptx_audit_passed": bool(audit_passed),
            "ptx_rank_audits": [
                {
                    "counts": item[0],
                    "passed": item[1]
                }
                for item in audit_results
            ],
            "energy_trace": [
                {"step": step, "energy": energy}
                for step, energy in overlapped["energy_trace"]
            ],
            "residual_trace": [
                {"step": step, "l2_residual": residual}
                for step, residual in overlapped["residual_trace"]
            ]
        }

        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)

        print("submission_manifest.json generated.", flush=True)

        if not manifest["cfl_stable"]:
            print("WARNING: CFL stability check failed.")

        if not manifest["parity_passed"]:
            print("WARNING: Serialized/overlapped parity failed.")

        if not manifest["ptx_audit_passed"]:
            print("WARNING: PTX audit failed.")

import os, sys, json, time, hashlib, platform, threading, subprocess, argparse
import numpy as np
from numba import njit

try:
    import resource
except ImportError:
    resource = None

NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)


def detect_cache_line_size():
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    try:
        with open(path, encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        if sys.platform == "darwin":
            try:
                return int(subprocess.check_output(
                    ["sysctl", "-n", "hw.cachelinesize"], text=True).strip())
            except Exception:
                pass
        return 64


def get_cpu_model():
    parts = [platform.machine(), platform.processor(), platform.uname().processor]
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(["wmic", "cpu", "get", "Name"], text=True)
            parts.extend(x.strip() for x in out.splitlines()[1:] if x.strip())
        except Exception:
            pass
    return "::".join(x for x in parts if x) or "unknown-cpu"


def available_cpus():
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    return list(range(os.cpu_count() or 1))


def pin_this_thread(cpu):
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {int(cpu)})
            return True
        except OSError:
            return False
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.GetCurrentThread.restype = ctypes.c_void_p
            k.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            k.SetThreadAffinityMask.restype = ctypes.c_size_t
            return bool(k.SetThreadAffinityMask(k.GetCurrentThread(), ctypes.c_size_t(1 << int(cpu))))
        except Exception:
            return False
    return False


def aligned_array(length, alignment=64):
    raw = np.zeros(length * 8 + alignment, dtype=np.uint8)
    offset = (-raw.ctypes.data) % alignment
    arr = np.ndarray((length,), dtype=np.uint64, buffer=raw, offset=offset)
    assert arr.ctypes.data % alignment == 0
    return arr, raw


@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, increment):
    increment = np.uint64(increment)
    for _ in range(iters):
        buffer[slot] = buffer[slot] + increment


@njit(nogil=True)
def worker_private_accumulate(iters, increment):
    increment = np.uint64(increment)
    value = np.uint64(0)
    for _ in range(iters):
        value += increment
    return value


def context_switches():
    if resource is None:
        return {"voluntary": None, "involuntary": None}
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {"voluntary": int(r.ru_nvcsw), "involuntary": int(r.ru_nivcsw)}


def run_workers(mode, array, stride, iterations, increments, cpus):
    barrier = threading.Barrier(NUM_THREADS)
    values, affinity, errors = [None]*4, [False]*4, []
    def target(tid):
        try:
            affinity[tid] = pin_this_thread(cpus[tid])
            barrier.wait()
            if mode == "private":
                values[tid] = int(worker_private_accumulate(iterations, np.uint64(increments[tid])))
            else:
                slot = tid * stride
                worker_strided_accumulate(array, slot, iterations, np.uint64(increments[tid]))
                values[tid] = int(array[slot])
        except BaseException as e:
            errors.append(repr(e))
    before = context_switches()
    start = time.perf_counter_ns()
    threads = [threading.Thread(target=target, args=(i,)) for i in range(NUM_THREADS)]
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.perf_counter_ns() - start
    after = context_switches()
    if errors:
        raise RuntimeError("; ".join(errors))
    delta = {k: None if before[k] is None else after[k]-before[k] for k in before}
    return values, elapsed, delta, affinity


def init_system_context(username):
    if not username or username == "REPLACE_WITH_YOUR_GIT_USERNAME":
        raise ValueError("Set your real GitHub username.")
    logical = os.cpu_count() or 1
    if logical < NUM_THREADS:
        raise RuntimeError(f"Need at least 4 logical CPUs; found {logical}")
    line = detect_cache_line_size()
    model = get_cpu_model()
    cpu_sig = f"{model}::{logical}"
    seed = int(hashlib.sha256(f"{username.strip()}::{cpu_sig}".encode()).hexdigest()[:16], 16)
    cpus = available_cpus()
    if len(cpus) < NUM_THREADS:
        raise RuntimeError(f"Only {len(cpus)} CPUs available to this process")
    return {"git_username": username.strip(), "logical_cores": logical,
            "cpu_model": model, "cache_line_size": line, "instance_seed": seed,
            "available_cpu_ids": cpus}


def run_stride_benchmark(context, iterations):
    cpus = context["available_cpu_ids"][:NUM_THREADS]
    seeds = [int((context["instance_seed"] + i*0x9E3779B97F4A7C15) & ((1<<64)-1))
             for i in range(NUM_THREADS)]
    increments = [1 + (seed % 7) for seed in seeds]
    expected = [iterations * increment for increment in increments]
    results = {}
    for stride in STRIDES:
        arr, backing = aligned_array((NUM_THREADS-1)*stride + 1, context["cache_line_size"])
        worker_strided_accumulate(arr, 0, 1, np.uint64(increments[0]))  # compile before timing
        arr[:] = 0
        values, elapsed, switches, affinity = run_workers(
            "shared", arr, stride, iterations, increments, cpus)
        if values != expected:
            raise AssertionError(f"Parity failed for stride {stride}: {values} != {expected}")
        results[str(stride)] = {
            "elapsed_ns": int(elapsed),
            "average_ns_per_update": float(elapsed/(NUM_THREADS*iterations)),
            "final_counters": values, "expected_counters": expected,
            "parity_passed": values == expected,
            "context_switches": switches, "affinity_applied": affinity
        }

    worker_private_accumulate(1, np.uint64(increments[0]))  # compile before timing
    values, elapsed, switches, affinity = run_workers(
        "private", None, 1, iterations, increments, cpus)
    if values != expected:
        raise AssertionError("Private baseline parity failed")
    results["register_private"] = {
        "elapsed_ns": int(elapsed),
        "average_ns_per_update": float(elapsed/(NUM_THREADS*iterations)),
        "final_counters": values, "expected_counters": expected,
        "parity_passed": values == expected,
        "context_switches": switches, "affinity_applied": affinity
    }
    for key in ("1", "4", "16", "register_private"):
        results[key]["slowdown_vs_stride_8"] = results[key]["elapsed_ns"]/results["8"]["elapsed_ns"]
    return results


def audit_llvm():
    sample, backing = aligned_array(4, 64)
    worker_strided_accumulate(sample, 0, 2, 3)
    ir = worker_strided_accumulate.inspect_llvm()
    ir_text = "\\n".join(str(v) for v in ir.values()) if isinstance(ir, dict) else str(ir)
    lines = [line for line in ir_text.splitlines()
             if "load i64" in line or "store i64" in line or "GIL" in line or "gil" in line.lower()]
    native_store = any("store i64" in line for line in lines)
    python_fallback = "PyObject" in ir_text or "pyobject" in ir_text.lower()
    return {"native_i64_store_present": native_store,
            "python_object_fallback_detected": python_fallback,
            "llvm_ir_excerpt": "\\n".join(lines)[:12000]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--username", default=os.environ.get("GIT_USERNAME", "Luri1337"))
    p.add_argument("--iterations", type=int, default=ITERATIONS_PER_THREAD,
                   help="Use a small value only for a smoke test; final run requires 60000000.")
    args = p.parse_args()
    if args.iterations <= 0: p.error("--iterations must be positive")

    ctx = init_system_context(args.username)
    print("===== SYSTEM CONTEXT =====")
    print(json.dumps({k:v for k,v in ctx.items() if k != "available_cpu_ids"}, indent=2))
    print("Available CPU IDs:", ctx["available_cpu_ids"])
    if args.iterations != ITERATIONS_PER_THREAD:
        print("WARNING: smoke test only; rerun without --iterations for submission.")
    benchmark = run_stride_benchmark(ctx, args.iterations)
    llvm = audit_llvm()
    parity = all(v["parity_passed"] for v in benchmark.values())
    llvm_pass = llvm["native_i64_store_present"] and not llvm["python_object_fallback_detected"]
    manifest = {**{k:v for k,v in ctx.items() if k != "available_cpu_ids"},
                "available_cpu_ids":ctx["available_cpu_ids"],
                "iterations_per_thread":args.iterations, "benchmark":benchmark,
                "llvm_audit":llvm, "parity_passed":parity,
                "llvm_audit_passed":llvm_pass,
                "verification_passed":bool(parity and llvm_pass)}
    print("\\n===== BENCHMARK RESULTS =====")
    for name, r in benchmark.items():
        print(f"{name:>14}: {r['elapsed_ns']:,} ns | {r['average_ns_per_update']:.6f} ns/update | "
              f"context switches={r['context_switches']} | affinity={r['affinity_applied']}")
    print("\\n===== LLVM AUDIT =====")
    print(json.dumps({k:v for k,v in llvm.items() if k != "llvm_ir_excerpt"}, indent=2))
    print(llvm["llvm_ir_excerpt"] or "(No matching LLVM load/store lines found)")
    print("\\nVerification passed:", manifest["verification_passed"])
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    print("cache_lab_manifest.json generated.")


if __name__ == "__main__":
    main()

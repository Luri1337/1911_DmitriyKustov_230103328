# Cache Line & False Sharing Mechanics in Multi-Core Python
**Student:** Dmitriy Kustov   
**Platform:** Windows 11, AMD Ryzen 5 5600H, 12 logical CPUs  
**Cache-line size used by the script:** 64 bytes  
**Threads:** 4  
**Iterations per thread:** 60,000,000

## Experimental results

The run completed all parity checks: every stride layout and the register-private baseline produced the expected final counters.

| Layout | Stride | Elapsed time (ns) | Reported average (ns/update) | Parity |
|---|---:|---:|---:|---|
| Contiguous | 1 | 2,164,600 | 0.009019 | PASS |
| Half-line | 4 | 965,300 | 0.004022 | PASS |
| Line-padded | 8 | 2,905,400 | 0.012106 | PASS |
| Prefetch-safe | 16 | 1,203,200 | 0.005013 | PASS |
| Register-private | — | 2,769,700 | 0.011540 | PASS |

Expected and actual counters matched in all five cases:

`[360000000,  180000000,  300000000,  120000000]`

The raw ratio of stride 1 to stride 8 elapsed time is:

`2,164,600 / 2,905,400 = 0.7450`

Thus, in this run stride 1 was about 25.5% faster than stride 8; it did **not** show a slowdown. The context-switch counters were unavailable on this Windows run and are recorded as `null`. Thread affinity was successfully applied for all four workers in each run.

**Important measurement limitation:** these timings are not physically credible for 240 million shared-memory read-modify-write operations: the reported average is below 0.013 ns per update, far below the time of a single CPU cycle. The LLVM audit also found `python_object_fallback_detected: true`, and the manifest reports `llvm_audit_passed: false` and `verification_passed: false`. The LLVM excerpt contains a promoted load (`%.174.promoted`) followed by a store, which is consistent with the compiler keeping the loop-carried value in a register and writing it back rather than issuing a memory read-modify-write on every iteration. Therefore, the measurements above are included as the program's observed output, but they must **not** be interpreted as valid evidence of false-sharing performance. The benchmark needs a corrected kernel that forces each iteration's memory access before the experiment can support quantitative conclusions.

## Question 1 — MESI Protocol State Transitions

With `stride = 1`, the four 8-byte counters occupy the same 64-byte cache line. Although Core 0 writes `buffer[0]` and Core 1 writes the different address `buffer[1]`, cache coherence operates on the whole line.

A simplified sequence is:

1. Core 0 obtains the line in an exclusive/modified ownership state before writing. Its store makes the line **Modified (M)** in Core 0.
2. Core 1 requests write ownership for its store. It issues a Read-For-Ownership (RFO) request for the same cache line.
3. The coherence protocol invalidates Core 0's copy. Core 0 must supply or write back the latest line as required by the protocol; Core 1 obtains ownership and enters **Modified (M)**.
4. When Core 0 writes again, it must request ownership back. The line is invalidated in Core 1, and ownership moves again.

The line repeatedly moves between cores. The store buffer cannot retire the store as a globally completed write until the core has obtained the required ownership/coherence permission. The RFO and invalidation acknowledgements therefore delay progress. This repeated ownership transfer is the false-sharing effect.

The exact transient states and whether an intermediate Shared (S) state is observed depend on the processor's coherence implementation. The essential behavior is repeated invalidation and exclusive ownership transfer for one shared cache line.

## Question 2 — Quantitative Slowdown & Coherency Overhead

The measured elapsed times and reported averages are listed in the table above. The measured stride-1/stride-8 ratio is **0.7450**, meaning stride 1 was faster in this particular output rather than slower.

This result contradicts the expected false-sharing pattern and is not a valid performance conclusion. The reported average values are physically implausible, and the LLVM audit failed. The likely explanation is compiler optimization/register promotion: the benchmark did not force a real shared-memory read-modify-write for every loop iteration. Consequently, these values should be treated as diagnostic output, not as measured cache-coherency costs.

## Question 3 — L2 Spatial Prefetcher & 128-Byte Pairing

With a 64-byte cache line, `stride = 8` places adjacent 8-byte counters 64 bytes apart. This removes direct sharing of the same cache line: each thread's counter begins on a different line, assuming the base array is correctly aligned.

Some processors use adjacent-line or spatial prefetching that fetches a neighboring 64-byte line together with the requested line, effectively working with a 128-byte pair or sector. As a result, two independently written 64-byte lines can still interact through prefetch traffic and cache-resource pressure even though they are not the same coherence line. The exact behavior is microarchitecture-dependent; prefetching a neighboring line does not by itself mean that a write to one line invalidates the other line.

With `stride = 16`, counters are 128 bytes apart. This separates them across the adjacent-line pair and is intended to reduce possible interference from such prefetch behavior.

In this run, stride 8 took **2,905,400 ns**, while stride 16 took **1,203,200 ns**. However, because the kernel's LLVM audit failed and the timings are implausibly small, this difference cannot be attributed confidently to the L2 adjacent-line prefetcher.

## Question 4 — Compiler Register Promotion vs. Volatile Memory Stores

An optimizing compiler may recognize that a thread is the only writer of its own counter during the loop. It can load the counter once, keep the changing value in a CPU register, perform repeated additions there, and store the final value back once. This transformation is called register promotion.

The LLVM excerpt from this run includes:

```llvm
%.174.promoted = load i64, ptr %.174, align 8
store i64 %2, ptr %.174, align 8
```

The presence of a native `i64` store confirms that native integer memory operations are generated somewhere in the compiled function. It does **not** prove that every loop iteration performs a memory store. The promoted-load name is evidence that the loop-carried value may have been moved into a register. The audit reported `native_i64_store_present: true`, but also `python_object_fallback_detected: true`; therefore the overall LLVM audit failed.

For a valid false-sharing experiment, the implementation must prevent this optimization and demonstrate in the generated LLVM/native code that the shared array element is loaded and stored on every iteration. The current output does not establish that requirement.

## Question 5 — CPython GIL Masking vs. `nogil=True` Execution

In ordinary CPython, only the thread holding the Global Interpreter Lock (GIL) can execute Python bytecode at a time. Multiple Python threads therefore do not continuously execute their counter-update bytecode in parallel on separate cores. They contend for the GIL, and increasing the thread count can add lock switching and scheduling overhead.

Because the bytecode updates are serialized, the experiment does not create the same simultaneous multi-core writes to a shared cache line. Hardware false sharing is therefore masked by the GIL.

Numba's `@njit(nogil=True)` compiles the numerical kernel to native code and releases the GIL while the kernel executes. This allows worker threads to run concurrently on different cores, making hardware cache-line contention observable—provided the compiler emits the intended memory access on every iteration. In the current run, the LLVM audit indicates that this final condition has not yet been verified.

## Verification status and conclusion

- Counter parity: **PASS**
- CPU affinity: **4/4 workers pinned**
- Context-switch telemetry: **unavailable (`null`) on this Windows run**
- Native `i64` store found: **true**
- Python-object fallback detector: **true**
- LLVM audit: **FAIL**
- Overall manifest verification: **FAIL**

The run confirms correct final counter values, but it does not yet demonstrate a valid false-sharing benchmark. The unexpectedly low elapsed times and failed LLVM audit indicate that the kernel must be corrected so each iteration performs a real memory read-modify-write. Do not present the current timing values as proof that stride 1 is faster or that false sharing is absent.

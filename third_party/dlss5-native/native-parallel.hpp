// SPDX-License-Identifier: MIT
// Copyright (c) 2026 the dlss5-linux contributors
//
// Deterministic data-parallel helpers for the native CPU path. The CPU path
// is the reference: every parallel loop splits an INDEPENDENT outer index
// (tokens, output rows, pixels, windows) while each element's inner
// accumulation keeps the single-threaded key order, so results are
// BIT-IDENTICAL to the serial reference at any thread count. Parallelism is
// opt-in (thread count 1 = the plain serial loop, zero threads spawned and
// zero allocations); spawning threads allocates thread stacks at the seam,
// which is outside the allocation-free kernel discipline's reference path.
#ifndef DLSS5_NR_NATIVE_PARALLEL_HPP
#define DLSS5_NR_NATIVE_PARALLEL_HPP

#include <algorithm>
#include <atomic>
#include <system_error>
#include <thread>

namespace dlss5_nr {

namespace parallel_detail {

inline std::atomic<int>& thread_count_state() {
    static std::atomic<int> count{1};
    return count;
}

}  // namespace parallel_detail

// The process-wide parallel thread count (1 = serial reference; the demo's
// 0 = hardware concurrency). Setting it is not thread-safe against its own
// use; do it at startup or between runs.
inline void set_parallel_threads(int count) {
    if (count <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        parallel_detail::thread_count_state().store(
            hw == 0 ? 1 : static_cast<int>(hw));
        return;
    }
    parallel_detail::thread_count_state().store(count);
}

inline int parallel_threads() {
    return parallel_detail::thread_count_state().load();
}

// Runs fn(i) for every i in [begin, end), split into contiguous chunks across
// the configured threads. The split changes WHO runs an index, never the
// order within one index's inner arithmetic. fn must not throw.
template <typename Idx, typename Fn>
inline void parallel_for(Idx begin, Idx end, Fn&& fn) {
    const Idx total = end - begin;
    const int threads = parallel_threads();
    if (total <= static_cast<Idx>(0)) {
        return;
    }
    if (threads <= 1 || total < static_cast<Idx>(2)) {
        for (Idx i = begin; i < end; ++i) {
            fn(i);
        }
        return;
    }
    const int workers = std::min(threads, static_cast<int>(
        total > static_cast<Idx>(64) ? 64 : total));
    const Idx chunk = (total + workers - 1) / workers;
    // Spawn-and-join: per-call thread cost is negligible against the
    // millisecond-scale kernel rows this drives. If spawning fails
    // (std::system_error), the unspawned chunks run inline and the spawned
    // threads are joined before returning: the seam never throws into the
    // no-throw kernel discipline and never terminates on a joinable
    // destructor.
    std::thread pool[64];
    int spawned = 0;
    Idx inline_lo = end;
    Idx inline_hi = end;
    for (int t = 0; t < workers; ++t) {
        const Idx lo = begin + static_cast<Idx>(t) * chunk;
        const Idx hi = lo + chunk < end ? lo + chunk : end;
        if (lo >= hi) {
            break;
        }
        try {
            pool[spawned] = std::thread([&fn, lo, hi]() {
                for (Idx i = lo; i < hi; ++i) {
                    fn(i);
                }
            });
            ++spawned;
        } catch (const std::system_error&) {
            inline_lo = lo;
            inline_hi = end;
            break;
        }
    }
    for (int t = 0; t < spawned; ++t) {
        pool[t].join();
    }
    for (Idx i = inline_lo; i < inline_hi; ++i) {
        fn(i);
    }
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_PARALLEL_HPP

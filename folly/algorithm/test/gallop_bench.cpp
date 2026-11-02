/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <folly/algorithm/gallop.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

/// Simulates a cursor walking a sorted posting list: repeatedly search forward
/// from the current position for a target a fixed stride ahead, then move the
/// cursor there. This is the shape every real caller has.
///
/// The variants sweep the two gallop_options knobs against std::lower_bound,
/// all through the public entry points so that what is measured is what
/// callers get, adapters included.
/// {32, 32} probes first at the linear-scan threshold, so a short stride is
/// resolved entirely by the tail; {0, 1} is a pure bisection with no tail; the
/// rest isolate which of the two knobs a given regime is sensitive to. The
/// stride sweep at n=100k walks the crossover, locating where galloping stops
/// paying as a fraction of the range length.

namespace {

/// Strictly increasing, so lower_bound of an element's own value lands
/// exactly on that element and every variant is asked the identical question.
const std::vector<uint32_t>& data(size_t n) {
  static std::map<size_t, std::vector<uint32_t>> cache;
  auto it = cache.find(n);
  if (it != cache.end()) {
    return it->second;
  }
  std::mt19937 rng{20260831};
  std::uniform_int_distribution<uint32_t> gap(1, 4);
  std::vector<uint32_t> v(n);
  uint32_t x = 0;
  for (auto& e : v) {
    x += gap(rng);
    e = x;
  }
  return cache.emplace(n, std::move(v)).first->second;
}

template <class Search>
void sweep(unsigned iters, size_t n, size_t stride, Search search) {
  const auto& v = data(n);
  const uint32_t* const base = v.data();
  size_t pos = 0;
  size_t acc = 0;
  for (unsigned i = 0; i < iters; ++i) {
    size_t want = pos + stride;
    if (want >= n) {
      pos = 0;
      want = stride < n ? stride : 0;
    }
    const uint32_t target = v[want];
    acc += static_cast<size_t>(search(base + pos, base + n, target) - base);
    pos = want;
  }
  folly::doNotOptimizeAway(acc);
}

void std_lower_bound(unsigned iters, size_t n, size_t stride) {
  sweep(iters, n, stride, [](const uint32_t* f, const uint32_t* l, uint32_t t) {
    return std::lower_bound(f, l, t);
  });
}

void gallop(
    unsigned iters, size_t n, size_t stride, folly::gallop_options opts) {
  sweep(
      iters,
      n,
      stride,
      [opts](const uint32_t* f, const uint32_t* l, uint32_t t) {
        return folly::gallop_partition_point(
            f, l, [t](uint32_t e) { return e >= t; }, opts);
      });
}

/// The benchmark macros paste the function name into an identifier, so each
/// strategy needs its own name rather than an inline argument list.
void tail32_init32(unsigned iters, size_t n, size_t stride) {
  gallop(iters, n, stride, {.linear_scan_threshold = 32, .initial_step = 32});
}
void no_tail_init1(unsigned iters, size_t n, size_t stride) {
  gallop(iters, n, stride, {.linear_scan_threshold = 0, .initial_step = 1});
}
void tail32_init1(unsigned iters, size_t n, size_t stride) {
  gallop(iters, n, stride, {.linear_scan_threshold = 32, .initial_step = 1});
}
void tail16_init16(unsigned iters, size_t n, size_t stride) {
  gallop(iters, n, stride, {.linear_scan_threshold = 16, .initial_step = 16});
}
void tail64_init64(unsigned iters, size_t n, size_t stride) {
  gallop(iters, n, stride, {.linear_scan_threshold = 64, .initial_step = 64});
}

/// Default tuning reached through gallop_lower_bound rather than
/// gallop_partition_point, so the comparator and projection layer is measured
/// too. That is the entry point most callers use.
void lower_bound_default(unsigned iters, size_t n, size_t stride) {
  sweep(iters, n, stride, [](const uint32_t* f, const uint32_t* l, uint32_t t) {
    return folly::gallop_lower_bound(f, l, t);
  });
}

} // namespace

#define GALLOP_VARIANTS(n, stride, label)                               \
  BENCHMARK_NAMED_PARAM(std_lower_bound, label, n, stride)              \
  BENCHMARK_RELATIVE_NAMED_PARAM(tail32_init32, label, n, stride)       \
  BENCHMARK_RELATIVE_NAMED_PARAM(no_tail_init1, label, n, stride)       \
  BENCHMARK_RELATIVE_NAMED_PARAM(tail32_init1, label, n, stride)        \
  BENCHMARK_RELATIVE_NAMED_PARAM(tail16_init16, label, n, stride)       \
  BENCHMARK_RELATIVE_NAMED_PARAM(tail64_init64, label, n, stride)       \
  BENCHMARK_RELATIVE_NAMED_PARAM(lower_bound_default, label, n, stride) \
  BENCHMARK_DRAW_LINE();

/// 1k: small enough that std::lower_bound stays in cache, which is the least
/// favourable size for galloping.
GALLOP_VARIANTS(1000, 0, n1k_aligned)
GALLOP_VARIANTS(1000, 1, n1k_stride1)
GALLOP_VARIANTS(1000, 8, n1k_stride8)
GALLOP_VARIANTS(1000, 32, n1k_stride32)
GALLOP_VARIANTS(1000, 500, n1k_stride500)

/// 100k: out of L2, where cache-line locality should favor galloping most.
GALLOP_VARIANTS(100000, 0, n100k_aligned)
GALLOP_VARIANTS(100000, 1, n100k_stride1)
GALLOP_VARIANTS(100000, 8, n100k_stride8)
GALLOP_VARIANTS(100000, 316, n100k_stride316)
/// Crossover sweep: where does galloping stop paying, as a fraction of n?
GALLOP_VARIANTS(100000, 1562, n100k_stride_n64)
GALLOP_VARIANTS(100000, 6250, n100k_stride_n16)
GALLOP_VARIANTS(100000, 12500, n100k_stride_n8)
GALLOP_VARIANTS(100000, 25000, n100k_stride_n4)
GALLOP_VARIANTS(100000, 50000, n100k_stride50k)

/// 1M.
GALLOP_VARIANTS(1000000, 1, n1m_stride1)
GALLOP_VARIANTS(1000000, 8, n1m_stride8)
GALLOP_VARIANTS(1000000, 1000, n1m_stride1k)
GALLOP_VARIANTS(1000000, 500000, n1m_stride500k)

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}

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

#include <cstdio>

#include <folly/debugging/symbolizer/ElfCache.h>
#include <folly/memory/ReentrantAllocator.h>

int main(int argc, char**) {
#if FOLLY_HAVE_ELF
  using folly::symbolizer::SignalSafeElfCache;
  struct ComparisonCache : SignalSafeElfCache {
    using SignalSafeElfCache::Path;
  };
  if (argc == 1) {
    SignalSafeElfCache::warmUp();
  }
  std::fputs("comparison-begin\n", stderr);
  folly::reentrant_allocator<void> alloc{
      folly::reentrant_allocator_options().block_size_lg(16).large_size_lg(12)};
  ComparisonCache::Path lhs("a", 1, alloc);
  ComparisonCache::Path rhs("b", 1, alloc);
  const bool less = lhs < rhs;
  std::fputs("comparison-end\n", stderr);
  return less ? 0 : 1;
#else
  return 0;
#endif
}

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
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <random>
#include <utility>
#include <vector>

#include <folly/lang/Hint.h>
#include <folly/portability/GTest.h>

namespace {

constexpr size_t kExhaustiveMaxSize = 64;

/// Sorted inputs of length n, one per content pattern worth distinguishing.
/// Element values are spaced by 2 so that targets can also land strictly
/// between two elements, and the duplicate patterns exercise the difference
/// between lower_bound and upper_bound.
std::vector<std::vector<int>> sortedInputs(size_t n) {
  std::vector<int> distinct(n);
  std::vector<int> runs(n);
  for (size_t i = 0; i < n; ++i) {
    distinct[i] = 2 * static_cast<int>(i);
    runs[i] = 2 * static_cast<int>(i / 3);
  }
  return {distinct, std::vector<int>(n, 6), runs};
}

/// The whole contract is that galloping lands exactly where the standard
/// algorithms land, for every partitioned input, every target, and -- for the
/// hinted overloads -- every starting guess including ones that overshoot.
void expectMatchesStdExhaustively(folly::gallop_options opts) {
  for (size_t n = 0; n <= kExhaustiveMaxSize; ++n) {
    for (const auto& v : sortedInputs(n)) {
      const auto begin = v.begin();
      const auto end = v.end();
      for (int target = -2; target <= 2 * static_cast<int>(n) + 2; ++target) {
        const auto lower = std::lower_bound(begin, end, target) - begin;
        const auto upper = std::upper_bound(begin, end, target) - begin;
        const auto atLeast = [&](size_t i) { return v[i] >= target; };

        const auto gotLower =
            folly::gallop_lower_bound(begin, end, target, {}, {}, opts) - begin;
        const auto gotUpper =
            folly::gallop_upper_bound(begin, end, target, {}, {}, opts) - begin;
        const auto gotPoint =
            folly::gallop_partition_point(
                begin, end, [&](int e) { return e >= target; }, opts) -
            begin;
        const auto gotIndex =
            folly::gallop_partition_point_index(size_t{0}, n, atLeast, opts);

        ASSERT_EQ(gotLower, lower) << "n=" << n << " target=" << target;
        ASSERT_EQ(gotUpper, upper) << "n=" << n << " target=" << target;
        ASSERT_EQ(gotPoint, lower) << "n=" << n << " target=" << target;
        ASSERT_EQ(gotIndex, static_cast<size_t>(lower))
            << "n=" << n << " target=" << target;

        for (size_t h = 0; h <= n; ++h) {
          const auto hint = begin + static_cast<ptrdiff_t>(h);
          const auto hintedLower =
              folly::gallop_lower_bound(
                  begin, hint, end, target, {}, {}, opts) -
              begin;
          const auto hintedUpper =
              folly::gallop_upper_bound(
                  begin, hint, end, target, {}, {}, opts) -
              begin;
          const auto hintedIndex = folly::gallop_partition_point_index(
              size_t{0}, h, n, atLeast, opts);

          ASSERT_EQ(hintedLower, lower)
              << "n=" << n << " target=" << target << " hint=" << h;
          ASSERT_EQ(hintedUpper, upper)
              << "n=" << n << " target=" << target << " hint=" << h;
          ASSERT_EQ(hintedIndex, static_cast<size_t>(lower))
              << "n=" << n << " target=" << target << " hint=" << h;
        }
      }
    }
  }
}

/// A random access iterator whose operator* yields a prvalue, mirroring the
/// decoding proxy arrays this library is meant to serve, where reading an
/// element costs an unaligned load plus a byte swap and there is no stored
/// object to point at. Nothing here may bind a reference to an element.
class ProxyIterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = uint32_t;
  using difference_type = ptrdiff_t;
  using pointer = void;
  using reference = uint32_t;

  //  Deleted rather than absent: the model deliberately offers only what the
  //  search calls, so an algorithm change that reaches for one of these fails
  //  to compile here rather than passing unnoticed.
  ProxyIterator() = delete;
  ProxyIterator(const uint32_t* data, ptrdiff_t index)
      : data_{data}, index_{index} {}

  uint32_t operator*() const { return data_[index_]; }
  uint32_t operator[](difference_type n) const = delete;

  ProxyIterator& operator++() { return *this += 1; }
  ProxyIterator& operator--() = delete;
  ProxyIterator& operator+=(difference_type n) {
    index_ += n;
    return *this;
  }
  ProxyIterator& operator-=(difference_type n) { return *this += -n; }

  friend ProxyIterator operator+(ProxyIterator it, difference_type n) {
    return it += n;
  }
  friend ProxyIterator operator-(ProxyIterator it, difference_type n) {
    return it -= n;
  }
  friend difference_type operator-(ProxyIterator a, ProxyIterator b) {
    return a.index_ - b.index_;
  }
  friend bool operator==(ProxyIterator a, ProxyIterator b) {
    return a.index_ == b.index_;
  }
  friend bool operator!=(ProxyIterator a, ProxyIterator b) { return !(a == b); }

 private:
  const uint32_t* data_{nullptr};
  ptrdiff_t index_{0};
};

} // namespace

TEST(Gallop, MatchesStdExhaustivelyDefaults) {
  expectMatchesStdExhaustively({});
}

TEST(Gallop, MatchesStdExhaustivelyTightest) {
  // Threshold 1 removes the linear tail's cover, so every window boundary the
  // gallop computes is checked by a bisection of one or two elements.
  expectMatchesStdExhaustively({.linear_scan_threshold = 1, .initial_step = 1});
}

TEST(Gallop, MatchesStdExhaustivelyNoLinearTail) {
  expectMatchesStdExhaustively({.linear_scan_threshold = 0, .initial_step = 1});
}

TEST(Gallop, MatchesStdExhaustivelyFirstProbeAtThreshold) {
  // The {K, K} shape, where the first probe lands exactly at the point the
  // linear tail takes over.
  expectMatchesStdExhaustively(
      {.linear_scan_threshold = 32, .initial_step = 32});
}

TEST(Gallop, MatchesStdExhaustivelyThresholdFour) {
  expectMatchesStdExhaustively({.linear_scan_threshold = 4, .initial_step = 4});
}

TEST(Gallop, RuntimeTuningIsNotConstantFolded) {
  // The options are ordinary runtime values, so a tuning the compiler cannot
  // see through must still give identical results.
  size_t tail = 5;
  size_t step = 3;
  folly::compiler_must_not_predict(tail);
  folly::compiler_must_not_predict(step);
  expectMatchesStdExhaustively(
      {.linear_scan_threshold = tail, .initial_step = step});
}

/// Searching a range whose answer sits near its back: hint == last is the
/// supported way to gallop from the end, so pin it with its own case rather
/// than leaving it implicit in the exhaustive hint sweep. The comparison is
/// against the forward std:: answer, since the whole point is that the caller
/// gets forward semantics without the predicate inversion and .base()
/// adjustment a std::reverse_iterator search would need.
TEST(Gallop, HintAtLastGallopsBackwardFromTheEnd) {
  std::vector<int> v(5000);
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = 2 * static_cast<int>(i);
  }
  const auto begin = v.begin();
  const auto end = v.end();

  // Answers at 1, 2, 4, ... elements from the back, plus the two extremes.
  for (size_t back = 1; back <= v.size(); back *= 2) {
    const int target = v[v.size() - back];
    ASSERT_EQ(
        folly::gallop_lower_bound(begin, end, end, target),
        std::lower_bound(begin, end, target))
        << "back=" << back;
    ASSERT_EQ(
        folly::gallop_upper_bound(begin, end, end, target),
        std::upper_bound(begin, end, target))
        << "back=" << back;
  }

  // Past the last element, and before the first.
  EXPECT_EQ(folly::gallop_lower_bound(begin, end, end, v.back() + 1), end);
  EXPECT_EQ(folly::gallop_lower_bound(begin, end, end, v.front() - 1), begin);
}

TEST(Gallop, EmptyRange) {
  const std::vector<int> v;
  EXPECT_EQ(folly::gallop_lower_bound(v.begin(), v.end(), 5), v.end());
  EXPECT_EQ(folly::gallop_upper_bound(v.begin(), v.end(), 5), v.end());
  EXPECT_EQ(
      folly::gallop_lower_bound(v.begin(), v.begin(), v.end(), 5), v.end());
  EXPECT_EQ(
      folly::gallop_partition_point_index(
          size_t{0}, size_t{0}, [](size_t) { return true; }),
      size_t{0});
}

TEST(Gallop, LargeRandomRangesWithDuplicates) {
  std::mt19937 rng{1729};
  for (const size_t n : {size_t{1000}, size_t{100000}}) {
    std::vector<int> v(n);
    // Draw from a range narrower than n so duplicates are dense.
    std::uniform_int_distribution<int> dist(0, static_cast<int>(n) / 4);
    for (auto& e : v) {
      e = dist(rng);
    }
    std::sort(v.begin(), v.end());

    for (int trial = 0; trial < 2000; ++trial) {
      const int target = dist(rng);
      const size_t hint = std::uniform_int_distribution<size_t>(0, n)(rng);
      const auto hintIt = v.begin() + static_cast<ptrdiff_t>(hint);
      ASSERT_EQ(
          folly::gallop_lower_bound(v.begin(), v.end(), target),
          std::lower_bound(v.begin(), v.end(), target));
      ASSERT_EQ(
          folly::gallop_upper_bound(v.begin(), v.end(), target),
          std::upper_bound(v.begin(), v.end(), target));
      ASSERT_EQ(
          folly::gallop_lower_bound(v.begin(), hintIt, v.end(), target),
          std::lower_bound(v.begin(), v.end(), target));
      ASSERT_EQ(
          folly::gallop_upper_bound(v.begin(), hintIt, v.end(), target),
          std::upper_bound(v.begin(), v.end(), target));
    }
  }
}

TEST(Gallop, HeterogeneousComparison) {
  // int64_t target against uint64_t elements: the transparent default
  // comparator must not force the target through the element type.
  const std::vector<uint64_t> v{1, 5, 9, 20, 21, 100};
  const int64_t target = 20;
  EXPECT_EQ(
      folly::gallop_lower_bound(v.begin(), v.end(), target) - v.begin(), 3);
  EXPECT_EQ(
      folly::gallop_upper_bound(v.begin(), v.end(), target) - v.begin(), 4);
}

TEST(Gallop, Projection) {
  using Entry = std::pair<int, char>;
  const std::vector<Entry> v{{1, 'a'}, {4, 'b'}, {4, 'c'}, {9, 'd'}};
  const auto key = [](const Entry& e) { return e.first; };

  EXPECT_EQ(
      folly::gallop_lower_bound(v.begin(), v.end(), 4, std::less<>{}, key) -
          v.begin(),
      1);
  EXPECT_EQ(
      folly::gallop_upper_bound(v.begin(), v.end(), 4, std::less<>{}, key) -
          v.begin(),
      3);
}

TEST(Gallop, DescendingWithCustomComparator) {
  const std::vector<int> v{9, 7, 7, 3, 1};
  EXPECT_EQ(
      folly::gallop_lower_bound(v.begin(), v.end(), 7, std::greater<>{}) -
          v.begin(),
      1);
  EXPECT_EQ(
      folly::gallop_upper_bound(v.begin(), v.end(), 7, std::greater<>{}) -
          v.begin(),
      3);
}

TEST(Gallop, ProxyIteratorYieldingPrvalues) {
  std::vector<uint32_t> storage(500);
  for (size_t i = 0; i < storage.size(); ++i) {
    storage[i] = static_cast<uint32_t>(3 * i);
  }
  const ProxyIterator begin{storage.data(), 0};
  const ProxyIterator end{
      storage.data(), static_cast<ptrdiff_t>(storage.size())};

  for (uint32_t target = 0; target < 1502; ++target) {
    const auto expected =
        std::lower_bound(storage.begin(), storage.end(), target) -
        storage.begin();
    ASSERT_EQ(folly::gallop_lower_bound(begin, end, target) - begin, expected)
        << "target=" << target;
    ASSERT_EQ(
        folly::gallop_lower_bound(begin, begin + 400, end, target) - begin,
        expected)
        << "target=" << target;
  }
}

TEST(Gallop, IndexDomainWithAccessorFunction) {
  // The shape taken by a container with no iterator at all, just an accessor
  // and a count.
  const std::vector<int> storage{2, 4, 6, 8, 10, 12, 14};
  const auto at = [&](size_t i) { return storage[i]; };

  for (int target = 0; target <= 16; ++target) {
    const auto expected = static_cast<size_t>(
        std::lower_bound(storage.begin(), storage.end(), target) -
        storage.begin());
    ASSERT_EQ(
        folly::gallop_partition_point_index(
            size_t{0},
            storage.size(),
            [&](size_t i) { return at(i) >= target; }),
        expected)
        << "target=" << target;
  }
}

TEST(Gallop, Constexpr) {
  static constexpr std::array<int, 8> kData{1, 3, 5, 7, 9, 11, 13, 15};
  static_assert(
      folly::gallop_lower_bound(kData.begin(), kData.end(), 9) -
          kData.begin() ==
      4);
  static_assert(
      folly::gallop_upper_bound(kData.begin(), kData.end(), 9) -
          kData.begin() ==
      5);
  static_assert(
      folly::gallop_partition_point_index(
          size_t{0}, kData.size(), [](size_t i) { return kData[i] >= 9; }) ==
      4);
}

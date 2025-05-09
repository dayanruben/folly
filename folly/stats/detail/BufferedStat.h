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

#pragma once

#include <folly/Range.h>
#include <folly/SharedMutex.h>
#include <folly/stats/DigestBuilder.h>
#include <folly/stats/detail/SlidingWindow.h>

namespace folly {
namespace detail {

/*
 * BufferedStat keeps a clock and every time period, will merge data from a
 * DigestBuilder into a DigestT. Updates are made by the first appender after
 * the expiry, or can be made at read time by calling update().
 */
template <typename DigestT, typename ClockT>
class BufferedStat {
 public:
  using TimePoint = typename ClockT::time_point;

  BufferedStat() = delete;

  BufferedStat(
      typename ClockT::duration bufferDuration,
      size_t bufferSize,
      size_t digestSize);

  virtual ~BufferedStat() {}

  void append(double value, TimePoint now = ClockT::now());

  void flush();

 protected:
  // https://www.mail-archive.com/llvm-bugs@lists.llvm.org/msg18280.html
  // Wrap the time point in something with a noexcept constructor.
  class AtomicTimePoint {
   private:
    using Duration = typename TimePoint::duration;
    using TimePointRep = typename TimePoint::rep;

    std::atomic<TimePointRep> rep_;

   public:
    explicit AtomicTimePoint(TimePoint value)
        : rep_{value.time_since_epoch().count()} {}

    TimePoint load(std::memory_order order) const {
      return TimePoint(Duration(rep_.load(order)));
    }
    void store(TimePoint value, std::memory_order order) {
      rep_.store(value.time_since_epoch().count(), order);
    }
  };

  const typename ClockT::duration bufferDuration_;
  AtomicTimePoint expiry_;
  mutable SharedMutex mutex_;

  virtual void onNewDigest(
      DigestT digest,
      TimePoint newExpiry,
      TimePoint oldExpiry,
      const std::unique_lock<SharedMutex>& g) = 0;

  // Update digest if now > expiry
  std::unique_lock<SharedMutex> updateIfExpired(TimePoint now);

  // Update digest unconditionally
  std::unique_lock<SharedMutex> update();

 private:
  DigestBuilder<DigestT> digestBuilder_;

  // Controls how digest updates happen in doUpdate
  enum class UpdateMode {
    OnExpiry,
    Now,
  };

  // Update digest. If updateMode == UpdateMode::Now digest is updated
  // unconditionally, else digest is updated only if expiry has passed.
  void doUpdate(
      TimePoint now,
      const std::unique_lock<SharedMutex>& g,
      UpdateMode updateMode);

  TimePoint roundUp(TimePoint t);
};

/*
 * BufferedDigest is a BufferedStat that holds data in a single digest.
 */
template <typename DigestT, typename ClockT>
class BufferedDigest : public BufferedStat<DigestT, ClockT> {
 public:
  using TimePoint = typename ClockT::time_point;

  BufferedDigest(
      typename ClockT::duration bufferDuration,
      size_t bufferSize,
      size_t digestSize);

  DigestT get(TimePoint now = ClockT::now());

  void onNewDigest(
      DigestT digest,
      TimePoint newExpiry,
      TimePoint oldExpiry,
      const std::unique_lock<SharedMutex>& g) final;

 private:
  DigestT digest_;
};

/*
 * BufferedSlidingWindow is a BufferedStat that holds data in a SlidingWindow.
 */
template <typename DigestT, typename ClockT>
class BufferedSlidingWindow : public BufferedStat<DigestT, ClockT> {
 public:
  using TimePoint = typename ClockT::time_point;

  BufferedSlidingWindow(
      size_t nBuckets,
      typename ClockT::duration bufferDuration,
      size_t bufferSize,
      size_t digestSize);

  std::vector<DigestT> get(TimePoint now = ClockT::now());

  void onNewDigest(
      DigestT digest,
      TimePoint newExpiry,
      TimePoint oldExpiry,
      const std::unique_lock<SharedMutex>& g) final;

 private:
  SlidingWindow<DigestT> slidingWindow_;
};

/*
 * BufferedMultiSlidingWindow is a BufferedStat that holds data in a
 * SlidingWindow for each requested duration/nBuckets, plus an all-time digest.
 *
 * This is more efficient than using a BufferedDigest plus one
 * BufferedSlidingWindow for each window, since the buffer is shared.
 */
template <typename DigestT, typename ClockT>
class BufferedMultiSlidingWindow : public BufferedStat<DigestT, ClockT> {
 public:
  using TimePoint = typename ClockT::time_point;

  // Minimum granularity is in seconds, so we can buffer at least one second.
  using WindowDef = std::pair<std::chrono::seconds, size_t>;

  struct Digests {
    Digests(DigestT at, std::vector<std::vector<DigestT>> ws)
        : allTime(std::move(at)), windows(std::move(ws)) {}

    DigestT allTime;
    std::vector<std::vector<DigestT>> windows;
  };

  BufferedMultiSlidingWindow(
      Range<const WindowDef*> defs, size_t bufferSize, size_t digestSize);

  Digests get(TimePoint now = ClockT::now());

  void onNewDigest(
      DigestT digest,
      TimePoint newExpiry,
      TimePoint oldExpiry,
      const std::unique_lock<SharedMutex>& g) final;

 private:
  struct Window {
    Window(
        TimePoint firstExpiry,
        std::chrono::seconds bucketDur,
        size_t nBuckets,
        size_t digestSize);

    std::chrono::seconds bucketDuration;
    TimePoint expiry;
    // curBucket accumulates pending updates before the bucket expires and is
    // committed to slidingWindow.
    DigestT curBucket;
    SlidingWindow<DigestT> slidingWindow;
  };

  size_t digestSize_;
  DigestT allTime_;
  std::vector<Window> windows_;
};

} // namespace detail
} // namespace folly

#include <folly/stats/detail/BufferedStat-inl.h>

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

#include <folly/Executor.h>
#include <folly/Function.h>
#include <folly/python/AsyncioExecutor.h>
#include <folly/python/Weak.h>

namespace folly {
namespace python {

namespace executor_detail {
extern folly::Function<AsyncioExecutor*(bool)> get_running_executor;
extern folly::Function<int(PyObject*, AsyncioExecutor*)> set_executor_for_loop;
} // namespace executor_detail

folly::Executor* getExecutor();

// Returns -1 if an executor was already set for loop, 0 otherwise. A NULL
// executor clears the current executor (caller is responsible for freeing
// any existing executor).
int setExecutorForLoop(PyObject* loop, AsyncioExecutor* executor);

} // namespace python
} // namespace folly

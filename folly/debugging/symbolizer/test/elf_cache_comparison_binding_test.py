#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import subprocess
import unittest


class ElfCacheComparisonBindingTest(unittest.TestCase):
    def test_comparison_warmup(self) -> None:
        helper = os.environ["ELF_CACHE_COMPARISON"]
        with open(helper, "rb") as binary:
            contents = binary.read()
        sanitized = b"__asan_init" in contents or b"__tsan_init" in contents
        env = dict(os.environ)
        env.pop("LD_BIND_NOW", None)
        env.pop("LD_DEBUG", None)
        if not sanitized:
            env["LD_DEBUG"] = "bindings"

        for warm in (False, True):
            with self.subTest(warm=warm):
                args = [helper] if warm else [helper, "cold"]
                result = subprocess.run(
                    args,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
                stderr = result.stderr.decode("utf-8", errors="replace")
                self.assertEqual(result.returncode, 0, stderr)
                self.assertEqual(result.stdout, b"")
                _, begin, after = stderr.partition("comparison-begin\n")
                trace, end, _ = after.partition("comparison-end\n")
                self.assertTrue(begin, stderr)
                self.assertTrue(end, stderr)
                if not sanitized:
                    symbols = re.findall(r"normal symbol `([^']+)'", trace)
                    comparisons = [s for s in symbols if s in ("memcmp", "strlen")]
                    if warm:
                        self.assertEqual(comparisons, [], stderr)
                    else:
                        self.assertIn("memcmp", comparisons, stderr)

# Copyright 2025 The Newton Developers
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
# ==============================================================================
"""Tests for warp_util host utilities."""

import numpy as np
import warp as wp
from absl.testing import absltest

from mujoco_warp._src import warp_util
from mujoco_warp._src.warp_util import EventTracer
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import check_toolkit_driver
from mujoco_warp._src.warp_util import event_scope
from mujoco_warp._src.warp_util import scoped_mathdx_gemm_disabled


@wp.kernel
def _add_one(x: wp.array(dtype=float)):
  i = wp.tid()
  x[i] = x[i] + 1.0


class WarpUtilTest(absltest.TestCase):
  def setUp(self):
    # ensure a clean global tracer stack for each test
    warp_util._STACK = None

  def tearDown(self):
    warp_util._STACK = None

  def test_event_tracer_disabled_returns_empty_trace(self):
    with EventTracer(enabled=False) as tracer:
      self.assertEqual(tracer.trace(), {})

  def test_only_one_tracer_at_a_time(self):
    with EventTracer():
      with self.assertRaises(ValueError):
        EventTracer()

  def test_event_scope_passthrough_without_tracer(self):
    calls = []

    @event_scope
    def fn(x):
      calls.append(x)
      return x + 1

    # no active tracer -> function runs directly, nothing recorded
    self.assertEqual(fn(41), 42)
    self.assertEqual(calls, [41])

  def test_event_scope_records_events_when_traced(self):
    @event_scope
    def do_work():
      x = wp.zeros(4, dtype=float)
      wp.launch(_add_one, dim=4, inputs=[x])
      return x

    with EventTracer() as tracer:
      do_work()
      do_work()
      trace = tracer.trace()

    self.assertIn("do_work", trace)
    events, sub_trace = trace["do_work"]
    # two invocations -> two elapsed-time samples
    self.assertEqual(len(events), 2)
    self.assertEqual(sub_trace, {})

  def test_merge_empty_and_disjoint(self):
    self.assertEqual(warp_util._merge({}, {}), {})
    merged = warp_util._merge({}, {"a": ((), {})})
    self.assertEqual(set(merged), {"a"})

  def test_merge_same_keys_concatenates_events(self):
    a = {"k": ((1,), {})}
    b = {"k": ((2, 3), {})}
    merged = warp_util._merge(a, b)
    self.assertEqual(merged["k"][0], (1, 2, 3))

  def test_merge_incompatible_stacks_raises(self):
    with self.assertRaises(ValueError):
      warp_util._merge({"a": ((), {})}, {"b": ((), {})})

  def test_cache_kernel_caches_and_hashes_arg_kinds(self):
    calls = []

    @cache_kernel
    def build(arg):
      calls.append(arg)
      return object()

    r_int_1 = build(3)
    r_int_2 = build(3)  # cache hit -> same object, factory not called again
    self.assertIs(r_int_1, r_int_2)
    self.assertEqual(len(calls), 1)

    # list arg (hashed via tuple) and an object exposing `.size` (hashed via size)
    build([1, 2])
    build(np.zeros(5))
    self.assertEqual(len(calls), 3)

  def test_check_toolkit_driver_runs(self):
    # Should not raise on any backend; emits an informational warning on HIP.
    check_toolkit_driver()

  def test_scoped_mathdx_gemm_disabled_noop_when_not_disabling(self):
    with scoped_mathdx_gemm_disabled(disable=False) as ctx:
      self.assertIsNone(ctx._config)

  def test_scoped_mathdx_gemm_disabled_restores_config(self):
    config = getattr(wp, "config", None)
    if config is None or not hasattr(config, "enable_mathdx_gemm"):
      self.skipTest("warp build has no enable_mathdx_gemm config")
    prev = config.enable_mathdx_gemm
    with scoped_mathdx_gemm_disabled():
      self.assertFalse(config.enable_mathdx_gemm)
    self.assertEqual(config.enable_mathdx_gemm, prev)


if __name__ == "__main__":
  wp.init()
  absltest.main()

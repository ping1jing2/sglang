# Copyright 2023-2024 SGLang Team
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

from __future__ import annotations

import bisect
import gc
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Optional, Union, List

import torch


import functools
import inspect
from torch._dynamo.eval_frame import (
    _stance,
    DisableContext,
    innermost_fn,
    _maybe_set_eval_frame,
    _callback_from_stance,
    _is_skip_guard_eval_unsafe_stance)
from torch._C._dynamo.eval_frame import set_skip_guard_eval_unsafe


def disable_context_call(self, fn):
    # Earlier this code was in the base class _TorchDynamoContext. But we
    # moved it here to have better code organization. For disable, we just
    # want the callback to be None. We don't have to check trace_rules or
    # create any wrapper.
    fn = innermost_fn(fn)

    if isinstance(fn, torch.nn.Module):
        mod = fn
        new_mod = OptimizedModule(mod, self)
        new_mod._torchdynamo_orig_callable = mod.forward
        return new_mod

    if inspect.isclass(fn):
        # User has wrapped the class with compile/disable decorator. Apply
        # disable to init/call method.
        cls_obj = fn
        # Disable on init is useful for reconstruction of bytecodes where we
        # want to prevent Dynamo from tracing into the init function. Check
        # test_reconstruction in test_model_output.py.
        cls_obj.__init__ = self(cls_obj.__init__)
        cls_obj.__call__ = self(cls_obj.__call__)
        if issubclass(cls_obj, torch.nn.Module):
            # NN module variable tracker directly inlines the _call_impl. Disable it.
            cls_obj._call_impl = self(cls_obj._call_impl)
        return cls_obj

    assert callable(fn)

    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        prior = _maybe_set_eval_frame(_callback_from_stance(self.callback))
        prior_skip_guard_eval_unsafe = set_skip_guard_eval_unsafe(
            _is_skip_guard_eval_unsafe_stance()
        )
        try:
            if torch._dynamo.eval_frame.was_captured:
                # global execution_contexts
                DisableContext.execution_contexts[self.ident] = self


                if True:
                    # initialization order is important to avoid synchronization
                    DisableContext.compiled_function_args = args
                    DisableContext.compiled_function_kwargs = kwargs
                    DisableContext.compiled_function = fn
                    result = fn(*args, **kwargs)
                    return result

            return fn(*args, **kwargs)
        finally:
            _maybe_set_eval_frame(prior)
            set_skip_guard_eval_unsafe(prior_skip_guard_eval_unsafe)

    _fn._torchdynamo_disable = True  # type: ignore[attr-defined]

    # Save the function pointer to find the original callable while nesting
    # of decorators.
    _fn._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]

    return _fn


def patch_dynamo_context():
    setattr(torch._dynamo.eval_frame.DisableContext, "execution_contexts", None)
    torch._dynamo.eval_frame.DisableContext.execution_contexts = {}

# use context manager
original_disable_context_call = None

def patch_dynamo_context_call():
    global original_disable_context_call
    original_disable_context_call = torch._dynamo.eval_frame.DisableContext.__call__
    torch._dynamo.eval_frame.DisableContext.__call__ = disable_context_call

def restore_dynamo_context_call():
    global original_disable_context_call
    torch._dynamo.eval_frame.DisableContext.__call__ = original_disable_context_call
    original_disable_context_call = None

# Copyright 2025 SGLang Team
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
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Optional, Union, List

import torch


import functools
import inspect
from torch._dynamo.eval_frame import (
    _TorchDynamoContext,
    _stance,
    DisableContext,
    innermost_fn,
    _maybe_set_eval_frame,
    _callback_from_stance,
    _is_skip_guard_eval_unsafe_stance)
from torch._dynamo.decorators import (
    skip)
from torch._C._dynamo.eval_frame import set_skip_guard_eval_unsafe


def patch_dynamo_context():
    setattr(torch._dynamo.eval_frame.DisableContext, "execution_contexts", None)
    setattr(torch._dynamo.eval_frame.DisableContext, "compiled_function_args", None)
    setattr(torch._dynamo.eval_frame.DisableContext, "compiled_function_kwargs", None)
    setattr(torch._dynamo.eval_frame.DisableContext, "compiled_function", None)

    torch._dynamo.eval_frame.DisableContext.execution_contexts = {}

# use context manager
original_disable_context_call = None
original_disable = None
last_context = None
last_context_call_original = None
wrapped_fn = None
capture_mode = False

def decorators_disable(fn=None, recursive=True):
    """
    Decorator to disable TorchDynamo

    If recursive=True, Dynamo is completely skipped on the decorated function
    frame as well as the recursively invoked functions.

    If recursive=False, Dynamo skips frames associated with the function code,
    but still process recursively invoked frames.
    """
    if recursive:
        if fn is not None:
            fn = innermost_fn(fn)
            assert callable(fn)

            DisableContext.compiled_function = fn

            context = DisableContext()
            context_fn = context(fn)

            context_fn._torchdynamo_disable = True

            global wrapped_fn
            wrapped_fn = context_fn

            global last_context
            last_context = context

            return context_fn
        return DisableContext()
    else:
        return skip(fn)


def patch_dynamo_context_call():
    global original_disable
    original_disable = torch._dynamo.decorators.disable
    torch._dynamo.decorators.disable = decorators_disable

def restore_dynamo_context_call():
    global original_disable
    torch._dynamo.decorators.disable = original_disable
    original_disable = None

def patch_last_context():
    pass

def restore_last_context():
    global last_context
    global last_context_call_original
    last_context.__call__ = last_context_call_original
    last_context_call_original = None
    last_context = None

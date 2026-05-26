# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group


def _get_yield_funcs():
    """Lazily import ubatching yield helpers to avoid circular imports."""
    try:
        from vllm.v1.worker.ubatching import (
            yield_and_switch_from_comm_to_compute as _to_compute,
        )
        from vllm.v1.worker.ubatching import (
            yield_and_switch_from_compute_to_comm as _to_comm,
        )

        return _to_comm, _to_compute
    except Exception:
        def _noop_to_comm(schedule: str = "default"):
            return None

        def _noop_to_compute(schedule: str = "default"):
            return None

        return _noop_to_comm, _noop_to_compute


def _all_reduce_with_dbo_yields(
    input_: torch.Tensor, schedule: str
) -> torch.Tensor:
    """All-reduce body. When DBO is active, wraps the AR with yield points so
    that the kernel runs on the comm_stream and overlaps with the other
    micro-batch's compute. Otherwise behaves as a vanilla AR.
    """
    try:
        from vllm.v1.worker.ubatching import (
            is_ubatching_globally_enabled as _is_enabled,
        )
        if not _is_enabled():
            return get_tp_group().all_reduce(input_)
    except Exception:
        return get_tp_group().all_reduce(input_)

    to_comm, to_compute = _get_yield_funcs()
    to_comm(schedule=schedule)
    out = get_tp_group().all_reduce(input_)
    to_compute(schedule=schedule)
    return out


# When VLLM_DBO_AR_AS_CUSTOM_OP=1, register the AR as a torch custom op so
# torch.compile / dynamo treats it as an opaque call and does not constant-fold
# the runtime DBO check (which evaluates to False at trace time, baking in a
# vanilla-AR-only graph that bypasses DBO at runtime). Required to make DBO
# coexist with cudagraph_mode=PIECEWISE under vLLM v1's fullgraph=True path.
_AR_OP = None
if os.getenv("VLLM_DBO_AR_AS_CUSTOM_OP", "0") == "1":
    try:
        from vllm.utils.torch_utils import direct_register_custom_op

        def _ar_op_impl(input_: torch.Tensor, schedule: str) -> torch.Tensor:
            return _all_reduce_with_dbo_yields(input_, schedule)

        def _ar_op_fake(input_: torch.Tensor, schedule: str) -> torch.Tensor:
            return torch.empty_like(input_)

        direct_register_custom_op(
            op_name="vllm_dbo_all_reduce",
            op_func=_ar_op_impl,
            mutates_args=[],
            fake_impl=_ar_op_fake,
        )
        _AR_OP = torch.ops.vllm.vllm_dbo_all_reduce.default
    except Exception:
        _AR_OP = None


def tensor_model_parallel_all_reduce(
    input_: torch.Tensor,
    *,
    schedule: str = "default",
) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    When ubatching/DBO is enabled, insert yield points around all-reduce to
    overlap compute and communication across microbatches. When ubatching is
    inactive, this behaves exactly like vanilla all-reduce.

    With VLLM_DBO_AR_AS_CUSTOM_OP=1, the call is routed through a torch
    custom op so torch.compile/dynamo treats it as opaque (preserves runtime
    DBO check under fullgraph=True compilation).
    """
    if _AR_OP is not None:
        return _AR_OP(input_, schedule)
    return _all_reduce_with_dbo_yields(input_, schedule)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)

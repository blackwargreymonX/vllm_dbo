# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

from vllm.config import ParallelConfig
from vllm.v1.attention.backend import CommonAttentionMetadata


@dataclass
class UBatchSlice:
    request_slice: slice
    token_slice: slice

    def is_empty(self) -> bool:
        return (
            self.request_slice.start == self.request_slice.stop
            or self.token_slice.start == self.token_slice.stop
        )

    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start


UBatchSlices: TypeAlias = list[UBatchSlice]


def is_last_ubatch_empty(
    orig_num_tokens: int, padded_num_tokens: int, num_ubatches: int
) -> bool:
    return (padded_num_tokens // num_ubatches) * (num_ubatches - 1) >= orig_num_tokens


def check_ubatch_thresholds(
    config: ParallelConfig,
    num_tokens: int,
    uniform_decode: bool,
    max_prompt_len: int = 0,
    num_reqs: int = 0,
) -> bool:
    """Decide whether to enable DBO ubatching for this batch.

    For prefill, additionally gate on per-request length (max_prompt_len) and
    request count (num_reqs):
    - num_reqs<2 (bs=1) goes through the mid-request split path which does
      not benefit from DBO and tends to regress 50-100%.
    - max_prompt_len < threshold means the cell is too small for DBO setup
      overhead to amortize.

    Threshold defaults / env overrides:
        VLLM_DBO_PREFILL_MIN_PL: int, default 2048. Min max_prompt_len for
            prefill DBO to fire. Set 0 to disable the gate.
        VLLM_DBO_MIN_REQS: int, default 2. Min num_reqs in batch for DBO to
            fire. Set 0 to disable the gate.
    """
    if not config.use_ubatching:
        return False
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold
    # Keep prefill DBO behind an explicit env switch for backward
    # compatibility with earlier TBO experiments.
    if os.getenv("VLLM_EXPERIMENTAL_PREFILL_DBO", "0") != "1":
        return False
    # Gate bs<2 (mid-request split path doesn't benefit from DBO).
    min_reqs = int(os.getenv("VLLM_DBO_MIN_REQS", "2"))
    if min_reqs > 0 and num_reqs > 0 and num_reqs < min_reqs:
        return False
    # Gate per-request length: setup overhead dominates on short prompts.
    pl_threshold = int(os.getenv("VLLM_DBO_PREFILL_MIN_PL", "2048"))
    if pl_threshold > 0 and max_prompt_len > 0 and max_prompt_len < pl_threshold:
        return False
    return num_tokens >= config.dbo_prefill_token_threshold


# This pads the last ubatch slice out to the total number of tokens
# (num_tokens + padding) since we do `create_ubatch_slices` before applying DP padding.
def _pad_out_ubatch_slices(
    ubatch_slices: UBatchSlices, num_total_tokens: int, num_reqs_padded: int
) -> UBatchSlices:
    last_slice = ubatch_slices[-1]
    padded_last_request_slice = slice(last_slice.request_slice.start, num_reqs_padded)
    padded_last_token_slice = slice(last_slice.token_slice.start, num_total_tokens)

    return ubatch_slices[:-1] + [
        UBatchSlice(padded_last_request_slice, padded_last_token_slice)
    ]


def _is_uniform_ubatch(num_scheduled_tokens: np.ndarray) -> bool:
    return (
        len(num_scheduled_tokens) > 0
        and int(num_scheduled_tokens[0]) > 0
        and bool(np.all(num_scheduled_tokens == num_scheduled_tokens[0]))
    )


def _get_uniform_request_split_points(
    cu_num_tokens: np.ndarray,
    num_tokens_padded: int,
    num_ubatches: int,
) -> list[int]:
    """Choose split points that balance both padded tokens and real requests.

    Decode DBO often pads the global token count up to a cudagraph/DP-friendly
    size. Splitting only at ``num_tokens_padded / num_ubatches`` balances the
    padded compute shape but can leave the real requests wildly imbalanced
    (e.g. 33 real decodes padded to 64 becomes 32/1). For uniform decode-like
    batches every request has the same token count, so we can restrict split
    points to request boundaries and pick the boundary that minimizes the worst
    normalized deviation from both the padded-token target and request target.
    """
    num_reqs = len(cu_num_tokens) - 1
    split_points: list[int] = []
    prev_req_idx = 0

    for split_idx in range(1, num_ubatches):
        remaining_splits = num_ubatches - split_idx
        min_req_idx = prev_req_idx + 1
        max_req_idx = num_reqs - remaining_splits
        if min_req_idx > max_req_idx:
            break

        target_tokens = num_tokens_padded * split_idx / num_ubatches
        target_reqs = num_reqs * split_idx / num_ubatches

        best_req_idx = min_req_idx
        best_score = (float("inf"), float("inf"), float("inf"))
        for req_idx in range(min_req_idx, max_req_idx + 1):
            token_boundary = int(cu_num_tokens[req_idx])
            token_score = abs(token_boundary - target_tokens) / max(
                num_tokens_padded, 1
            )
            request_score = abs(req_idx - target_reqs) / max(num_reqs, 1)
            score = (
                max(token_score, request_score),
                token_score + request_score,
                token_score,
            )
            if score < best_score:
                best_score = score
                best_req_idx = req_idx

        split_points.append(int(cu_num_tokens[best_req_idx]))
        prev_req_idx = best_req_idx

    return split_points


def _get_token_split_points(
    num_tokens_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
) -> list[int]:
    if split_point is None:
        return [
            int(num_tokens_padded) * i // num_ubatches
            for i in range(1, num_ubatches)
        ]

    if isinstance(split_point, int):
        return [split_point * i for i in range(1, num_ubatches)]

    return split_point


def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
) -> tuple[UBatchSlices | None, UBatchSlices | None]:
    if not should_ubatch:
        return None, None

    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])
    num_tokens = int(cu_num_tokens[-1])

    if (
        split_point is None
        and len(num_scheduled_tokens) >= num_ubatches
        and _is_uniform_ubatch(num_scheduled_tokens)
    ):
        token_split_points = _get_uniform_request_split_points(
            cu_num_tokens, num_tokens_padded, num_ubatches
        )
    else:
        token_split_points = _get_token_split_points(
            num_tokens_padded, num_ubatches, split_point
        )

    ubatch_slices = []
    start_token = 0

    # Add the end point to the split points to make iteration easier
    all_points = token_split_points + [num_tokens]

    for end_token in all_points:
        token_slice = slice(start_token, end_token)

        # Determine request slices using exclusive stop semantics
        # Ubatch includes requests whose tokens overlap [start_token, end_token)

        # Start at the request that contains the start_token
        # or the request starting exactly at start_token (if on boundary)
        req_start = int(np.searchsorted(cu_num_tokens, start_token, side="right") - 1)

        # Stop at the request that starts at or after end_token
        req_stop = int(np.searchsorted(cu_num_tokens, end_token, side="left"))

        req_slice = slice(req_start, req_stop)
        ubatch_slices.append(UBatchSlice(req_slice, token_slice))

        start_token = end_token

    ubatch_slices_padded = _pad_out_ubatch_slices(
        ubatch_slices, num_tokens_padded, num_reqs_padded
    )

    assert sum(s.num_tokens for s in ubatch_slices_padded) == num_tokens_padded

    return ubatch_slices, ubatch_slices_padded


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """
    Creates a new query_start_loc that corresponds to the requests in
    request_slice.

    Note: This function creates a new tensor to hold the new query_start_locs.
    This will break cudagraph compatibility.
    """
    return (
        query_start_loc[request_slice.start : request_slice.stop + 1]
        - query_start_loc[request_slice.start]
    )


def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice, attn_metadata: CommonAttentionMetadata
) -> CommonAttentionMetadata:
    """
    This function creates a new CommonAttentionMetadata that corresponds to
    the requests included in ubatch_slice
    """

    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], (
        "Token slice start outside of first request"
    )
    # NOTE: last token can be outside of the last request if we have CG padding.

    # If the request is split across ubatches, we have to adjust the metadata.
    # splits_first_request: The first request in this slice is the continuation of
    #                       a request that started in a previous slice.
    # splits_last_request:  The last request in this slice continues into the
    #                       next slice.
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(
        attn_metadata.query_start_loc, request_slice
    )

    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, got {len(query_start_loc)}"
    )

    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped
    seq_lens = attn_metadata.seq_lens[request_slice]
    # Read raw fields to avoid triggering the deprecated D2H-syncing properties.
    seq_lens_cpu = (
        attn_metadata._seq_lens_cpu[request_slice]
        if attn_metadata._seq_lens_cpu is not None
        else None
    )
    seq_lens_cpu_upper_bound = (
        attn_metadata.seq_lens_cpu_upper_bound[request_slice]
        if attn_metadata.seq_lens_cpu_upper_bound is not None
        else None
    )
    num_computed_tokens_cpu = (
        attn_metadata._num_computed_tokens_cpu[request_slice]
        if attn_metadata._num_computed_tokens_cpu is not None
        else None
    )

    if splits_last_request:
        # NOTE: We use start_locs (the original query_start_loc_cpu) to calculate
        # the tokens skipped because query_start_loc_cpu might have been modified
        # if splits_first_request is True.
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped

        # Make sure we don't modify the seq_lens tensors
        #  (not cudagraph compatible)
        seq_lens = seq_lens.clone()
        seq_lens[-1] -= tokens_skipped
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu.clone()
            seq_lens_cpu[-1] -= tokens_skipped
        if seq_lens_cpu_upper_bound is not None:
            seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound.clone()
            seq_lens_cpu_upper_bound[-1] -= tokens_skipped

    assert seq_lens_cpu_upper_bound is not None
    max_seq_len = int(seq_lens_cpu_upper_bound.max())

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item()
    )

    # This is to account for the case where we are in a dummy
    # run and query_start_loc_cpu is full of 0s
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
    )


def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: CommonAttentionMetadata,
) -> list[CommonAttentionMetadata]:
    """
    Creates a new CommonAttentionMetadata instance that corresponds to the
    requests for each UBatchSlice in ubatch_slices.

    Note: This function does not modify common_attn_metadata
    """
    results = []
    for ubatch_slice in ubatch_slices:
        results.append(_make_metadata_with_slice(ubatch_slice, common_attn_metadata))

    return results

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import os

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

import torch

_buffer = None

deepep_check_threshold = 1e-7
deepep_check_threshold_probs = 1e-5
def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int, overlap_moe_expert_parallel_comm: bool = False, token_num: int = 0, hidden_size: int = 0, num_topk: int = 0) -> Buffer:
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        use_ace = os.getenv("USE_DEEPEP_ACE", "0") == "1"
        num_ace_buffers = 2 if overlap_moe_expert_parallel_comm else 1
        if use_ace:
            assert token_num > 0 and hidden_size > 0 and num_topk > 0, "token_num, hidden_size and num_topk must be provided when using ACE buffers."
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes, use_ace=use_ace, num_ace_buffers=num_ace_buffers, token_num=token_num, hidden_size=hidden_size, num_topk=num_topk)
        # _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes, use_ace=use_ace, num_ace_buffers=1, train_mode=True)

    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        overlap_moe_expert_parallel_comm=False,
        cur_chunk_id = None
    ):
        """Forward pass of fused dispatch."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x), overlap_moe_expert_parallel_comm, x.size(0), x.size(1), token_indices.size(1))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            ace_buffer_idx=0,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.overlap_moe_expert_parallel_comm = overlap_moe_expert_parallel_comm
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)
        ctx.cur_chunk_id = cur_chunk_id
        
        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, num_tokens_per_rank, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs_output, grad_tokens_per_expert, num_tokens_per_rank, grad_handle
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output), ctx.overlap_moe_expert_parallel_comm)
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        
        if os.environ.get("SYNC_DEEPEP", "0") == "1":
            # print(f"[sherry info] megatron/core/transformer/moe/fused_a2a.py SYNC_DEEPEP!!!")
            torch.cuda.synchronize()
            
        if os.environ.get("DEEPEP_RETURN_A2AV_OUTPUT", "0") == "1":
            grad_x, grad_token_probs, after_event, a2av_grad_output = buffer.combine(
                grad_output.contiguous(),
                handle,
                topk_weights=grad_token_probs_output.float(),
                previous_event=previous_event,
                async_finish=ctx.async_finish,
                allocate_on_comm_stream=ctx.allocate_on_comm_stream,
            )
        else:
            grad_x, grad_token_probs, after_event = buffer.combine(
                grad_output.contiguous(),
                handle,
                topk_weights=grad_token_probs_output.float(),
                previous_event=previous_event,
                async_finish=ctx.async_finish,
                allocate_on_comm_stream=ctx.allocate_on_comm_stream,
            )
            
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
            
        if os.environ.get("DUMP_DEEPEP", "0") == "1":
            from musa_patch.utils import add_global_debug_datas, get_iteration_id, get_run_batch_id, get_tensor_desc, try_dump_invalid_grad_tensor
            info = {
                "grad_output": get_tensor_desc(grad_output),
                "grad_token_probs_output": get_tensor_desc(grad_token_probs_output),
                "grad_x": get_tensor_desc(grad_x),
                "grad_token_probs": get_tensor_desc(grad_token_probs),
            }
            if os.environ.get("DEEPEP_RETURN_A2AV_OUTPUT", "0") == "1":
                info["a2av_grad_output"] = get_tensor_desc(a2av_grad_output)

            if (any([v["max"] > deepep_check_threshold for k, v in info.items() if "token_probs" not in k]) or
                any([v["max"] > deepep_check_threshold_probs for k, v in info.items() if "token_probs" in k])):
                iter_id = get_iteration_id()
                batch_id = get_run_batch_id()
                cur_chunk_id = ctx.cur_chunk_id
                save_key = add_global_debug_datas(iter_id, batch_id, f"chunk_{cur_chunk_id}_dispatch_backward_post_check", info)
                try_dump_invalid_grad_tensor(iter_id, batch_id, f"{save_key}.grad_output", grad_output, info["grad_output"]["max"])
                try_dump_invalid_grad_tensor(iter_id, batch_id, f"{save_key}.grad_token_probs_output", grad_token_probs_output, info["grad_token_probs_output"]["max"])
                try_dump_invalid_grad_tensor(iter_id, batch_id, f"{save_key}.grad_x", grad_x, info["grad_x"]["max"])
                try_dump_invalid_grad_tensor(iter_id, batch_id, f"{save_key}.grad_token_probs", grad_token_probs, info["grad_token_probs"]["max"])
                if os.environ.get("DEEPEP_RETURN_A2AV_OUTPUT", "0") == "1":
                    try_dump_invalid_grad_tensor(iter_id, batch_id, f"{save_key}.a2av_grad_output", a2av_grad_output, info["a2av_grad_output"]["max"])
                
            
        return grad_x, None, grad_token_probs, None, None, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False, overlap_moe_expert_parallel_comm=False,
        cur_chunk_id = None):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x), overlap_moe_expert_parallel_comm)
        if os.environ.get("DEEPEP_RETURN_A2AV_OUTPUT", "0") == "1":
            combined_x, _, after_event, a2av_output = buffer.combine(
                x,
                handle=handle,
                async_finish=async_finish,
                previous_event=previous_event,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        else:
            combined_x, _, after_event = buffer.combine(
                x,
                handle=handle,
                async_finish=async_finish,
                previous_event=previous_event,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.overlap_moe_expert_parallel_comm = overlap_moe_expert_parallel_comm
        ctx.cur_chunk_id = cur_chunk_id

        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        
        if os.environ.get("SYNC_DEEPEP", "0") == "1":
            torch.cuda.synchronize()
            
        ace_buffer_idx = 1 if ctx.overlap_moe_expert_parallel_comm else 0
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
            ace_buffer_idx=ace_buffer_idx,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
            
        if os.environ.get("DUMP_DEEPEP", "0") == "1":
            from musa_patch.utils import add_global_debug_datas, get_iteration_id, get_run_batch_id, get_tensor_desc, try_dump_invalid_grad_tensor
            info = {
                "grad_output": get_tensor_desc(grad_output),
                "grad_x": get_tensor_desc(grad_x)
            }
            if any([v["max"] > deepep_check_threshold for v in info.values()]):
                iter_id = get_iteration_id()
                batch_id = get_run_batch_id()
                cur_chunk_id = ctx.cur_chunk_id
                save_key = add_global_debug_datas(iter_id, batch_id, f"chunk_{cur_chunk_id}_combine_backward_post_check", info)
                try_dump_invalid_grad_tensor(iter_id, batch_id, f"{save_key}.grad_output", grad_output, info["grad_output"]["max"])
                try_dump_invalid_grad_tensor(iter_id, batch_id, f"{save_key}.grad_x", grad_x, info["grad_x"]["max"])
                
        return grad_x, None, None, None, None, None, None


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        overlap_moe_expert_parallel_comm = False,
        cur_chunk_id = None
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
            overlap_moe_expert_parallel_comm,
            cur_chunk_id
        )

    def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False, overlap_moe_expert_parallel_comm = False, cur_chunk_id = None):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream, overlap_moe_expert_parallel_comm, cur_chunk_id)

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP"""
        Buffer.set_num_sms(num_sms)

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None


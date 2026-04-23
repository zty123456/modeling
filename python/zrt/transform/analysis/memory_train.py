from __future__ import annotations

from dataclasses import dataclass

from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.transform.base import GraphPass
from zrt.transform.context import TransformContext


@dataclass
class TrainingMemoryBudget:
    """Training memory budget breakdown."""
    weights_mb: float
    grads_mb: float
    opt_state_mb: float
    activations_mb: float
    comm_buffer_mb: float
    offloaded_mb: float
    overhead_mb: float
    total_hbm_mb: float
    host_resident_mb: float
    capacity_mb: float
    is_feasible: bool


class TrainMemoryPass(GraphPass):
    """TrainMemoryPass for training memory analysis."""
    name = "memory_train"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run TrainMemoryPass on the graph.
        
        Args:
            graph: Input OpGraph
            ctx: TransformContext with training config
            
        Returns:
            New OpGraph with training memory annotations
        """
        g = graph.clone()
        if not ctx.training:
            return g
        
        # Get Zero sharding factors
        shards = g.metadata.get("zero", {
            "weight_shard": 1,
            "grad_shard": 1,
            "optstate_shard": 1
        })
        
        # Calculate parameters on rank
        params = self._params_on_rank(g, ctx)
        
        # Get data types from training config
        param_dtype = ctx.training.param_dtype if hasattr(ctx.training, "param_dtype") else "bf16"
        grad_dtype = ctx.training.grad_dtype if hasattr(ctx.training, "grad_dtype") else "fp32"
        
        # Calculate memory breakdown
        weight_bytes = params * self._dtype_bytes(param_dtype) / shards["weight_shard"]
        grad_bytes = params * self._dtype_bytes(grad_dtype) / shards["grad_shard"]
        opt_bytes = self._opt_state_total_bytes(ctx.training.optimizer, params) / shards["optstate_shard"]
        
        # Calculate activation memory
        act_bytes = self._activations_korthikanti(g, ctx)
        
        # Calculate communication buffer
        comm_bytes = self._comm_buffer(ctx)
        
        # Apply offload
        off = ctx.training.offload
        offloaded = 0.0
        if off.opt_state:
            offloaded += opt_bytes * off.pct
            opt_bytes *= (1 - off.pct)
        if off.grads:
            offloaded += grad_bytes * off.pct
            grad_bytes *= (1 - off.pct)
        if off.params:
            offloaded += weight_bytes * off.pct
            weight_bytes *= (1 - off.pct)
        
        # Convert to MB
        weights_mb = weight_bytes / (1024**2)
        grads_mb = grad_bytes / (1024**2)
        opt_state_mb = opt_bytes / (1024**2)
        activations_mb = act_bytes / (1024**2)
        comm_buffer_mb = comm_bytes / (1024**2)
        offloaded_mb = offloaded / (1024**2)
        
        # Calculate overhead (5% of total)
        total_without_overhead = weights_mb + grads_mb + opt_state_mb + activations_mb + comm_buffer_mb
        overhead_mb = total_without_overhead * 0.05
        
        # Calculate total HBM usage
        total_hbm_mb = total_without_overhead + overhead_mb
        host_resident_mb = offloaded_mb
        
        # Get memory capacity from hardware spec
        capacity_mb = ctx.hw_spec.memory.capacity_gb * 1024
        is_feasible = total_hbm_mb <= capacity_mb
        
        # Create memory budget
        budget = TrainingMemoryBudget(
            weights_mb=weights_mb,
            grads_mb=grads_mb,
            opt_state_mb=opt_state_mb,
            activations_mb=activations_mb,
            comm_buffer_mb=comm_buffer_mb,
            offloaded_mb=offloaded_mb,
            overhead_mb=overhead_mb,
            total_hbm_mb=total_hbm_mb,
            host_resident_mb=host_resident_mb,
            capacity_mb=capacity_mb,
            is_feasible=is_feasible
        )
        
        # Add memory budget to graph metadata
        g.metadata["train_memory"] = budget
        
        return g

    def _params_on_rank(self, graph: OpGraph, ctx: TransformContext) -> int:
        """Calculate parameters on rank.
        
        Args:
            graph: OpGraph to analyze
            ctx: TransformContext with parallel config
            
        Returns:
            Number of parameters on rank
        """
        # This is a simplified implementation
        # In practice, you would need to analyze the graph to count parameters
        
        # Get model profile from context if available
        if hasattr(ctx, 'profile') and ctx.profile:
            total_params = ctx.profile.param_count()
        else:
            # Estimate based on common model sizes
            total_params = 70e9  # 70B parameters as default
        
        # Apply parallelism sharding
        tp = ctx.parallel.tp
        pp = ctx.parallel.pp
        dp = ctx.parallel.dp
        cp = ctx.parallel.cp
        
        # Calculate params per rank
        params_per_rank = total_params / (tp * pp * dp * cp)
        
        return int(params_per_rank)

    def _dtype_bytes(self, dtype: str) -> int:
        """Get bytes per element for a dtype.
        
        Args:
            dtype: Data type string
            
        Returns:
            Bytes per element
        """
        dtype_map = {
            "fp16": 2,
            "bf16": 2,
            "fp32": 4,
            "fp64": 8,
            "int8": 1,
            "int16": 2,
            "int32": 4,
            "int64": 8,
        }
        return dtype_map.get(dtype.lower(), 2)  # Default to 2 bytes (bf16)

    def _opt_state_total_bytes(self, optimizer: str, params: int) -> int:
        """Calculate total optimizer state bytes.
        
        Args:
            optimizer: Optimizer name
            params: Number of parameters
            
        Returns:
            Total optimizer state bytes
        """
        if optimizer == "adam":
            # Adam: 8 bytes per parameter (2 * 4 bytes for momentums)
            return params * 8
        elif optimizer == "muon":
            # Muon: 4 bytes per parameter + scratch space
            return params * 4 + params * 2  # 2 bytes for scratch
        else:
            # Default: 4 bytes per parameter
            return params * 4

    def _activations_korthikanti(self, graph: OpGraph, ctx: TransformContext) -> int:
        """Calculate activation memory using Korthikanti's method.
        
        Args:
            graph: OpGraph to analyze
            ctx: TransformContext with training config
            
        Returns:
            Activation memory in bytes
        """
        # Korthikanti's formula: ~34 * bs * seq * h / tp
        # This is a simplified implementation
        
        # Get batch size and sequence length from context or metadata
        batch_size = ctx.training.micro_batch if hasattr(ctx.training, "micro_batch") else 1
        seq_len = 4096  # Default sequence length
        
        # Get hidden size from model profile if available
        hidden_size = 8192  # Default hidden size
        if hasattr(ctx, 'profile') and ctx.profile:
            hidden_size = ctx.profile.hidden_size
        
        # Apply parallelism factors
        tp = ctx.parallel.tp
        cp = ctx.parallel.cp
        
        # Calculate activation memory
        # Apply CP factor (1/cp on attention activation)
        cp_factor = 1.0 / cp if cp > 1 else 1.0
        
        # Apply recompute factor
        recompute_factor = 0.5  # Assuming half of activations are recomputed
        
        # Calculate activation bytes
        activation_bytes = 34 * batch_size * seq_len * hidden_size / tp * cp_factor * recompute_factor
        
        return int(activation_bytes)

    def _comm_buffer(self, ctx: TransformContext) -> int:
        """Calculate communication buffer size.
        
        Args:
            ctx: TransformContext with parallel config
            
        Returns:
            Communication buffer size in bytes
        """
        # Get hidden size from model profile if available
        hidden_size = 8192  # Default hidden size
        if hasattr(ctx, 'profile') and ctx.profile:
            hidden_size = ctx.profile.hidden_size
        
        # Calculate communication buffer for TP all-reduce
        tp = ctx.parallel.tp
        if tp <= 1:
            return 0
        
        # TP all-reduce needs hidden_size * 2 (for bf16) * 2 (two buffers) bytes
        return hidden_size * 2 * 2

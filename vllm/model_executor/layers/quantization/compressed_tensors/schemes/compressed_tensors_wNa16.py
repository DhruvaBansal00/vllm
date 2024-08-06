from typing import Callable, List, Optional

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.kernels import (
    MPLinearLayerConfig, choose_mp_linear_kernel)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_TYPES_MAP = {
    4: scalar_types.uint4b8,
    8: scalar_types.uint8b128,
}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())


class CompressedTensorsWNA16(CompressedTensorsScheme):

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 group_size: Optional[int] = None):

        self.pack_factor = 32 // num_bits
        self.strategy = strategy

        self.group_size: int
        if group_size is None:
            if self.strategy != "channel":
                raise ValueError(
                    "Marlin kernels require group quantization or "
                    "channelwise quantization, but found no group "
                    "size and strategy is not channelwise.")
            self.group_size = -1
        else:
            self.group_size = group_size

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {WNA16_SUPPORTED_TYPES_MAP.keys()}")

        self.quant_type = WNA16_SUPPORTED_TYPES_MAP[num_bits]

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(self, layer: torch.nn.Module, output_size: int,
                       input_size: int, output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        output_size_per_partition = sum(output_partition_sizes)

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=\
                (input_size_per_partition, output_size_per_partition),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=False,
            act_reordering=False
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        # If group_size is -1, we are in channelwise case.
        channelwise = (self.group_size == -1)
        group_size = input_size if channelwise else self.group_size
        row_parallel = (input_size != input_size_per_partition)
        # In the case of channelwise quantization, we need to replicate the
        # scales across all gpus.
        partition_scales = (row_parallel and not channelwise)

        weight_scale_dim = None
        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            weight_scale_dim = 1
            scales_and_zp_size = input_size_per_partition // group_size

        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

        set_weight_attrs(
            weight, {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.pack_factor,
                "weight_loader": weight_loader
            })

        weight_scale = Parameter(
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        set_weight_attrs(
            weight_scale, {
                "weight_loader": weight_loader,
                "input_dim": weight_scale_dim,
                "output_dim": 0
            })

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = Parameter(torch.empty(2, dtype=torch.int64),
                                 requires_grad=False)

        set_weight_attrs(weight_shape, {
            "weight_loader": weight_loader,
            "ignore_warning": True,
        })

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        self.kernel = kernel_type(mp_linear_kernel_config,
                                  w_q_param_name="weight_packed",
                                  w_s_param_name="weight_scale",
                                  w_zp_param_name=None,
                                  w_gidx_param_name=None)

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # convert `weight_packed` from:
        #  {input_dim = 1, output_dim = 0, packed_dim = 1}
        # to:
        #  {input_dim = 0, output_dim = 1, packed_dim = 0}
        # expected the kernels `process_weights_after_loading`
        replace_parameter(layer, "weight_packed", layer.weight_packed.t())

        # convert `weight_scale` from:
        #  {input_dim = 1, output_dim = 0}
        # to:
        #  {input_dim = 0, output_dim = 1}
        # expected the kernels `process_weights_after_loading`
        replace_parameter(layer, "weight_scale", layer.weight_scale.t())

        self.kernel.process_weights_after_loading(layer)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)

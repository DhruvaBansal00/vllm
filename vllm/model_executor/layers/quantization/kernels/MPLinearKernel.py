from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.scalar_type import ScalarType


@dataclass
class MPLinearLayerConfig:
    full_weight_shape: Tuple[int, int]  # [in, out]
    partition_weight_shape: Tuple[int, int]
    weight_type: ScalarType
    act_type: torch.dtype
    group_size: int
    zero_points: bool
    act_reordering: bool


class MPLinearKernel(ABC):

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError

    def __init__(self,
                 c: MPLinearLayerConfig,
                 w_q_param_name: str,
                 w_s_param_name: str,
                 w_zp_param_name: Optional[str] = None,
                 w_gidx_param_name: Optional[str] = None) -> None:
        assert self.can_implement(c)
        self.config = c
        self.w_q_name = w_q_param_name
        self.w_s_name = w_s_param_name
        self.w_zp_name = w_zp_param_name
        self.w_gidx_name = w_gidx_param_name

    # note assumes that
    #  `getattr(layer, w_q_name)` is:
    #     {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `getattr(layer, w_s_name)` is:
    #     {input_dim = 0, output_dim = 1}
    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def _transform_param(self, layer: torch.nn.Module, name: Optional[str],
                         fn: Callable) -> None:
        if name is not None and getattr(layer, name, None) is not None:
            replace_parameter(layer, name, fn(getattr(layer, name)))

    def _get_weight_params(
            self, layer: torch.nn.Module
    ) -> Tuple[torch.Tensor,  # w_q
               torch.Tensor,  # w_s
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        return (
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
            getattr(layer, self.w_zp_name or "", None),
            getattr(layer, self.w_gidx_name or "", None),
        )

# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

import math
import re

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

from matplotlib import use
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import PeftConfig, PeftType
from peft.utils import transpose


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    
    
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_nums: int = field(default=None, metadata={"help": "Numbers of Lora"})
    modal_list: List[str] = field(default=None, metadata={"help": "List of modalities"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False,
        metadata={"help": "Merge weights of the original model and the Lora model"},
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"
        },
    )
    moe_type: str = field(
        default="lora",
    )

    continue_training: list[str] = field(
        default=None, metadata={"help": "List of modalities to continue training"}
    )

    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    use_orthogonal_loss: bool = field(
        default=False, metadata={"help": "Use orthogonal loss for Lora"}
    )
    
    topk : int = field(default=1, metadata={"help": "topk for LoraMoE"})
    
    expert_nums: int = field(default=1, metadata={"help": "Numbers of experts"})

    def __post_init__(self):
        self.peft_type = PeftType.LORA


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):  # LoraConfig, CasualLM
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        # loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        # loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        # if loaded_in_4bit or loaded_in_8bit:
        #     raise ImportError(
        #         "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
        #         "You can install it with `pip install bitsandbytes`."
        #     )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (
                self.peft_config.merge_weights or self.peft_config.inference_mode
            )
            and not is_hf_device_map_available,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(
                    key.endswith(target_key)
                    for target_key in self.peft_config.target_modules
                )
            if target_module_found:  # here
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                if isinstance(target, torch.nn.Linear):
                    if self.peft_config.moe_type == "lora":
                        kwargs.update({"use_orthogonal_loss": self.peft_config.use_orthogonal_loss})
                        new_module = Lora_Linear(
                            target.in_features, target.out_features, bias=bias, **kwargs
                        )

                    elif self.peft_config.moe_type == "lora_moe":
                        kwargs.update({"modal_list": self.peft_config.modal_list})
                        kwargs.update(
                            {"continue_training": self.peft_config.continue_training}
                        )
                        kwargs.update({"lora_nums": self.peft_config.lora_nums})
                        # kwargs.update({"expert_nums": self.peft_config.expert_nums})
                        new_module = MultiModalLora(
                            target.in_features, target.out_features, bias=bias, **kwargs
                        )
                    elif self.peft_config.moe_type == "lora_moe_topk":
                        kwargs.update({"modal_list": self.peft_config.modal_list})
                        kwargs.update(
                            {"continue_training": self.peft_config.continue_training}
                        )
                        kwargs.update({"lora_nums": self.peft_config.lora_nums})
                        kwargs.update({"expert_nums": self.peft_config.expert_nums})
                        kwargs.update({"topk": self.peft_config.topk})
                        new_module = MultiModalLoRAMoE(
                            target.in_features, target.out_features, bias=bias, **kwargs
                        )
                    elif self.peft_config.moe_type == "lora_moe_mg":
                        kwargs.update({"modal_list": self.peft_config.modal_list})
                        
                        kwargs.update({"lora_nums": self.peft_config.lora_nums})
                        
                        kwargs.update({"topk": self.peft_config.topk})
                        new_module = MultiModalLoRAMoE_MG(
                            target.in_features, target.out_features, bias=bias, **kwargs
                        )
                    
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {
            k: v.value if isinstance(v, Enum) else v
            for k, v in asdict(self.peft_config).items()
        }
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Lora_Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        use_orthogonal_loss: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)

            self.scaling = self.lora_alpha / r
            # if use_rslora:
            #     self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
            # else:
            #     self.scaling[adapter_name] = lora_alpha / r

            if use_orthogonal_loss:
                Vr, Sr, Ur = torch.svd_lowrank(self.weight, self.r, niter=2)
                Uhr = Ur.t()
                self.lora_refer_B = torch.diag(torch.sqrt(Sr)) @ Uhr
                self.lora_refer_A = Vr @ torch.diag(torch.sqrt(Sr))

                self.lora_refer_A = nn.Parameter(self.lora_refer_A, requires_grad=True)
                self.lora_refer_B = nn.Parameter(self.lora_refer_B, requires_grad=True)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
            raise ImportError(":(")
        elif self.r > 0 and not self.merged:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

            if self.r > 0:
                result = (
                    result
                    + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
                )

        return result


class Lora_MoE(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.lora_num = lora_nums

        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
            for i in range(self.lora_num):
                setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
                setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, "lora_A0"):
            for i in range(self.lora_num):
                nn.init.kaiming_uniform_(
                    getattr(self, f"lora_A{i}").weight, a=math.sqrt(5)
                )
                nn.init.zeros_(getattr(self, f"lora_B{i}").weight)

            nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_route.train(mode)
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").train(mode)
            getattr(self, f"lora_B{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        self.lora_route.eval()
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").eval()
            getattr(self, f"lora_B{i}").eval()

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x: torch.Tensor, task_types=None):
        if self.disable_adapters:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
            raise ImportError(":(")
        elif self.r > 0 and not self.merged:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

            if self.r > 0:
                route_weight = nn.functional.softmax(
                    self.lora_route(x), dim=-1, dtype=torch.float32
                ).to(result.dtype)

                for i in range(self.lora_num):
                    result = (
                        result
                        + torch.unsqueeze(route_weight[:, :, i], -1)
                        * getattr(self, f"lora_B{i}")(
                            getattr(self, f"lora_A{i}")(self.lora_dropout(x))
                        )
                        * self.scaling
                    )

        return result


def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: torch.Tensor = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

class MultiModalLora(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        modal_list: List[str] = None,
        continue_training: list[str] = None,
        **kwargs,
    ):
        self.modal_list = modal_list
        assert len(modal_list) == lora_nums

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.lora_nums = lora_nums

        self.fan_in_fan_out = fan_in_fan_out

        self.continue_training = continue_training

        self.num_experts_per_tok = 2
        self.in_features = in_features
        self.out_features = out_features

        # Actual trainable parameters
        if r > 0:
            if continue_training is not None:
                self.lora_route_old = nn.Linear(
                    in_features, self.lora_nums - len(continue_training), bias=False
                )
                self.lora_route_new = nn.Linear(in_features, self.lora_nums, bias=False)
                self.weight_scale = WeightScale(lora_nums)
            else:
                self.lora_route = nn.Linear(in_features, self.lora_nums, bias=False)
            for i in self.modal_list:
                setattr(self, f"lora_A_{i}", nn.Linear(in_features, r, bias=False))
                setattr(self, f"lora_B_{i}", nn.Linear(r, out_features, bias=False))

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, f"lora_A_{self.modal_list[0]}"):
            for i in self.modal_list:
                nn.init.kaiming_uniform_(
                    getattr(self, f"lora_A_{i}").weight, a=math.sqrt(5)
                )
                nn.init.zeros_(getattr(self, f"lora_B_{i}").weight)
            if self.continue_training is not None:
                nn.init.zeros_(self.lora_route_new.weight)
            else:
                nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))
        if hasattr(self, "weight_scale"):
            nn.init.kaiming_uniform_(
                self.weight_scale.weight_scale[0].weight, a=math.sqrt(5)
            )
            nn.init.zeros_(self.weight_scale.weight_scale[2].weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.continue_training is not None:
            self.lora_route_new.train(mode)
            self.weight_scale.train(mode)
        else:
            self.lora_route.train(mode)
        for i in self.modal_list:
            getattr(self, f"lora_A_{i}").train(mode)
            getattr(self, f"lora_B_{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        if self.continue_training is not None:
            self.lora_route_new.eval()
            self.weight_scale.eval()
        else:
            self.lora_route.eval()
        for i in self.modal_list:
            getattr(self, f"lora_A_{i}").eval()
            getattr(self, f"lora_B_{i}").eval()

    def forward(self, x: torch.Tensor, modal_types="image"):
        

        if self.disable_adapters:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
            raise ImportError(":(")
        elif self.r > 0 and not self.merged:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

            if self.r > 0:
                if self.continue_training is not None:
                    if modal_types in self.continue_training:
                        
                        route_logits = self.lora_route_new(x).to(result.dtype)
                        route_weight = nn.functional.softmax(
                            route_logits, dim=-1, dtype=torch.float32
                        ).to(result.dtype)
                        route_weight = route_weight + self.weight_scale(route_weight)
                    else:
                        route_weight_old = nn.functional.softmax(
                            self.lora_route_old(x), dim=-1, dtype=torch.float32
                        ).to(result.dtype)
                        route_weight_new = torch.zeros(
                            route_weight_old.shape[:-1] + (len(self.continue_training),)
                        ).to(
                            dtype=route_weight_old.dtype, device=route_weight_old.device
                        )
                        route_weight = torch.cat(
                            (route_weight_old, route_weight_new), dim=-1
                        )
                else:
                    route_weight = nn.functional.softmax(
                        self.lora_route(x), dim=-1, dtype=torch.float32
                    ).to(result.dtype)

                for i, m in enumerate(self.modal_list):
                    result = (
                        result
                        + torch.unsqueeze(route_weight[:, :, i], -1)
                        * getattr(self, f"lora_B_{m}")(
                            getattr(self, f"lora_A_{m}")(self.lora_dropout(x))
                        )
                        * self.scaling
                    )

            return result


class WeightScale(nn.Module):
    def __init__(self, num_local_experts):
        super().__init__()
        self.weight_scale = nn.Sequential(
            nn.Linear(num_local_experts, num_local_experts * 2, bias=False),
            nn.GELU(),
            nn.Linear(num_local_experts * 2, num_local_experts, bias=False),
        )

    def forward(self, hidden_states):
        return self.weight_scale(hidden_states)

class MultiModalLoRAMoE(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        expert_nums: int = 1,
        lora_alpha: int = 1,
        lora_nums: int = 1,
        topk: int =1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        modal_list: List[str] = None,
        continue_training: list[str] = None,
        **kwargs,
    ):
        self.modal_list = modal_list
        assert expert_nums == len(modal_list) * lora_nums

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.lora_nums = lora_nums
        
        self.expert_nums = expert_nums
        
        self.lora_A = nn.ModuleDict()
        
        self.lora_B = nn.ModuleDict()

        self.fan_in_fan_out = fan_in_fan_out

        self.continue_training = continue_training

        self.num_experts_per_tok = topk
        self.in_features = in_features
        self.out_features = out_features
        
        self.l_aux = 0.0

        # Actual trainable parameters
        if r > 0:
            if continue_training is not None:
                self.lora_route_old = nn.Linear(
                    in_features, self.lora_nums - len(continue_training), bias=False
                )
                self.lora_route_new = nn.Linear(in_features, self.lora_nums, bias=False)
                self.weight_scale = WeightScale(lora_nums)
                nn.init.zeros_(self.lora_route_new.weight)
            else:
                self.lora_route = nn.Linear(in_features, self.expert_nums, bias=False)
                nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))
            for i in self.modal_list:
                for j in range(lora_nums):
                    self.lora_A[f'{i}_{j}'] = nn.Linear(in_features, r, bias=False)
                    self.lora_B[f'{i}_{j}'] = nn.Linear(r, out_features, bias=False)
                    
                    nn.init.kaiming_uniform_(self.lora_A[f'{i}_{j}'].weight,a=math.sqrt(5))
                    nn.init.zeros_(self.lora_B[f'{i}_{j}'].weight)
                    
                # setattr(self, f"lora_A_{i}", nn.Linear(in_features, r, bias=False))
                # setattr(self, f"lora_B_{i}", nn.Linear(r, out_features, bias=False))

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, "weight_scale"):
            nn.init.kaiming_uniform_(
                self.weight_scale.weight_scale[0].weight, a=math.sqrt(5)
            )
            nn.init.zeros_(self.weight_scale.weight_scale[2].weight)

    def forward(self, x: torch.Tensor, modal_types="image"):
        
        
        ori_result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        oshape = ori_result.shape
        
        ishape = x.shape
        inputs = x.view(-1,ishape[-1])
        gate_logits = self.lora_route(inputs)
        # l_aud
        l_aux =0.0
        if self.expert_nums>1:
            gates = F.softmax(gate_logits, dim=1)
            indices1_s = torch.argmax(gates, dim=1)
            num_experts = int(gates.shape[1])
            
            mask1 = F.one_hot(indices1_s, num_classes=num_experts)

            # Compute l_aux
            me = torch.mean(gates, dim=0)
            ce = torch.mean(mask1.float(), dim=0)
            l_aux = torch.mean(me * ce) * num_experts * num_experts
        self.l_aux = l_aux
        # print("laux",l_aux)
        
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(ori_result.view(-1, oshape[-1]))
        for i, (expert_A, expert_B) in enumerate(zip(self.lora_A,self.lora_B)):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * self.lora_B[expert_B](
                            self.lora_A[expert_A](self.lora_dropout(inputs[batch_idx]))
                        ) * self.scaling
        
        results_out = results.view(oshape)
        
        results_out = results_out + ori_result
        
        return results_out
    

class MultiModalLoRAMoE_MG(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 1,
        topk: int =1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        modal_list: List[str] = None,
        **kwargs,
    ):
        self.modal_list = modal_list
        # assert expert_nums == len(modal_list) * lora_nums

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.lora_nums = lora_nums
        
        self.lora_A = nn.ModuleDict()
        
        self.lora_B = nn.ModuleDict()
        
        self.lora_route = nn.ModuleDict()
        
        self.scaling=nn.ParameterDict()

        self.fan_in_fan_out = fan_in_fan_out

        self.num_experts_per_tok = topk
        self.in_features = in_features
        self.out_features = out_features
        
        self.l_aux = 0.0

        # Actual trainable parameters
        if r > 0:

            for i in self.modal_list:
                self.lora_route[i] = nn.Linear(in_features, self.lora_nums, bias=False)
                nn.init.kaiming_uniform_(self.lora_route[i].weight, a=math.sqrt(5))
                
                self.lora_A[i]=nn.ModuleDict()
                self.lora_B[i]=nn.ModuleDict()
                
                for j in range(lora_nums):
                    self.lora_A[i][str(j)] = nn.Linear(in_features, r, bias=False)
                    self.lora_B[i][str(j)] = nn.Linear(r, out_features, bias=False)
                    
                    nn.init.kaiming_uniform_(self.lora_A[i][str(j)].weight,a=math.sqrt(5))
                    nn.init.zeros_(self.lora_B[i][str(j)].weight)
                    
                self.scaling[i] = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        # if hasattr(self, f"lora_A_{self.modal_list[0]}"):
        #     for i in self.modal_list:
        #         nn.init.kaiming_uniform_(
        #             getattr(self, f"lora_A_{i}").weight, a=math.sqrt(5)
        #         )
        #         nn.init.zeros_(getattr(self, f"lora_B_{i}").weight)
        #     if self.continue_training is not None:
        #         nn.init.zeros_(self.lora_route_new.weight)
        #     else:
        #         nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))
        if hasattr(self, "weight_scale"):
            nn.init.kaiming_uniform_(
                self.weight_scale.weight_scale[0].weight, a=math.sqrt(5)
            )
            nn.init.zeros_(self.weight_scale.weight_scale[2].weight)

    def forward(self, x: torch.Tensor, modal="image"):
        
        ori_result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        oshape = ori_result.shape
        
        if modal not in self.modal_list:
            return ori_result
        
        ishape = x.shape
        inputs = x.view(-1,ishape[-1])
        gate_logits = self.lora_route[modal](inputs)
        # l_aud
        l_aux =0.0
        if self.lora_nums>1:
            gates = F.softmax(gate_logits, dim=1)
            indices1_s = torch.argmax(gates, dim=1)
            num_experts = int(gates.shape[1])
            
            mask1 = F.one_hot(indices1_s, num_classes=num_experts)

            # Compute l_aux
            me = torch.mean(gates, dim=0)
            ce = torch.mean(mask1.float(), dim=0)
            l_aux = torch.mean(me * ce) * num_experts * num_experts
        self.l_aux = l_aux
        # print("laux",l_aux)
        
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(ori_result.view(-1, oshape[-1]))
        
        
        
        for i, (expert_A, expert_B) in enumerate(zip(self.lora_A[modal],self.lora_B[modal])):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * self.lora_B[modal][expert_B](
                            self.lora_A[modal][expert_A](self.lora_dropout(inputs[batch_idx]))
                        ) * self.scaling[modal]
        
        results_out = results.view(oshape)
        
        results_out = results_out + ori_result
        
        return results_out
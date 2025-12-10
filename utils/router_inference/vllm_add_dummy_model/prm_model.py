import os
import torch
import re
import sys
import vllm
from torch import nn
from vllm.model_executor.layers.pooler import (
    Optional,
    List,
    PoolerConfig,
    PoolingType,
    PoolingMetadata,
    PoolingTensors,
    PoolingSequenceGroupOutput,
    PoolerOutput
)
from vllm.model_executor.models.qwen2_rm import (
    Qwen2Model,
    AutoWeightsLoader,
    VllmConfig,
    maybe_prefix,
    IntermediateTensors,
    PoolingMetadata,
    PoolingType,
    PoolerOutput,
    AttentionMetadata,
    Iterable,
    Union,
    Tuple,
    SupportsPP
)


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use.
        normalize: Whether to normalize the pooled data.
    """

    def __init__(
            self,
            pooling_type: PoolingType,

    ):
        super().__init__()

        self.pooling_type = pooling_type

    @classmethod
    def from_config_with_defaults(
            cls,
            pooler_config: PoolerConfig,
            pooling_type: PoolingType,
    ) -> Optional["Pooler"]:
        if pooler_config is None:
            return None
        return cls(
            pooling_type=PoolingType[pooler_config.pooling_type]
            if pooler_config.pooling_type is not None else pooling_type,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """Pools specific information from hidden states based on metadata."""

        prompt_lens = PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

        if self.pooling_type == PoolingType.ALL:
            offset = 0
            pooled_data = []
            for prompt_len in prompt_lens:
                pooled_data.append(hidden_states[offset:offset + prompt_len])
                offset += prompt_len
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        pooled_outputs = [
            PoolingSequenceGroupOutput(data.flatten()) for data in pooled_data
        ]

        return PoolerOutput(outputs=pooled_outputs)


class Qwen2ForTokenClassification(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config
        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                             "supported. This model uses sliding window "
                             "but `max_window_layers` = {} is less than "
                             "`num_hidden_layers` = {}. Please open an issue "
                             "to discuss this feature.".format(
                config.max_window_layers,
                config.num_hidden_layers,
            ))

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        self.score = nn.Linear(config.hidden_size, config.num_labels)

        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.ALL)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        logits = self.score(hidden_states)
        return logits

    def pooler(
            self,
            hidden_states: torch.Tensor,
            pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self,
                                   ignore_unexpected_prefixes=["lm_head."])
        loader.load_weights(weights)


def register():
    from vllm import ModelRegistry
    if "Qwen2ForTokenClassification" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Qwen2ForTokenClassification",
                                     "vllm_add_dummy_model.prm_model:Qwen2ForTokenClassification")

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import os
from typing import Optional

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import habitat_llm
from habitat_llm.llm.base_llm import BaseLLM


def __getattr__(name):
    if name == "AnthropicClaude":
        from habitat_llm.llm.anthropic_claude import AnthropicClaude

        return AnthropicClaude
    if name == "BedrockClaude":
        from habitat_llm.llm.bedrock_claude import BedrockClaude

        return BedrockClaude
    if name == "OpenAIChat":
        from habitat_llm.llm.openai_chat import OpenAIChat

        return OpenAIChat
    raise AttributeError(name)


def instantiate_llm(
    llm_name: str, generation_params: Optional[DictConfig] = None, **kwargs
):
    """
    Creates a new LLM instance based on the config.
    :param llm_name: name of the llm you want to intantiate, file must exist in conf/llm/{llm_name}.yaml
    :generation_params: config with parameters
    """
    if generation_params is None:
        generation_params = {}

    # Get the path to the LLM config file
    habitat_llm_dir_path = os.path.dirname(habitat_llm.__file__)
    llm_config_path = f"{habitat_llm_dir_path}/conf/llm/{llm_name}.yaml"
    assert os.path.exists(
        llm_config_path
    ), f"LLM config file not found at {llm_config_path}"

    # Load the LLM config file
    llm_config = OmegaConf.load(llm_config_path)

    # Update the config with the kwargs
    if generation_params:
        llm_config.generation_params = OmegaConf.merge(
            llm_config.generation_params, OmegaConf.create(generation_params)
        )

    if kwargs:
        llm_config = OmegaConf.merge(llm_config, OmegaConf.create(kwargs))

    llm = instantiate(llm_config.llm)(llm_config)

    return llm

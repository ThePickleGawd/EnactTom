#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

# Suppress verbose httpx logs from OpenAI client
logging.getLogger("httpx").setLevel(logging.WARNING)

from enacttom.api_costs import maybe_append_usage_event
from habitat_llm.llm.base_llm import BaseLLM, Prompt

# Load .env file if it exists (for API keys)
_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        # Fallback: manually parse .env if dotenv not installed
        with open(_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def generate_message(multimodal_prompt, image_detail="auto"):
    # Converts the multimodal prompt to the OpenAI format.
    content = []
    for prompt_type, prompt_value in multimodal_prompt:
        if prompt_type == "text":
            message_item = {"type": "text", "text": prompt_value}
        else:
            message_item = {
                "type": "image_url",
                "image_url": {
                    "url": prompt_value,
                    "detail": image_detail,
                },
            }
        content.append(message_item)
    return {"role": "user", "content": content}


class OpenAIChat(BaseLLM):
    """
    LLM implementation using OpenAI's Chat API.
    Uses environment variable: OPENAI_API_KEY

    Supports model aliases for convenience:
        gpt5, gpt-5           -> gpt-5
        gpt5-mini, gpt-5-mini -> gpt-5-mini
        gpt5.1, gpt-5.1       -> gpt-5.1
        gpt5.2, gpt-5.2       -> gpt-5.2
    """

    # Model aliases mapping short names to full OpenAI model IDs
    MODEL_ALIASES: Dict[str, str] = {
        # GPT-5
        "gpt5": "gpt-5",
        # GPT-5 Mini
        "gpt5-mini": "gpt-5-mini",
        # GPT-5.1
        "gpt5.1": "gpt-5.1",
        # GPT-5.2
        "gpt5.2": "gpt-5.2",
        # Fireworks-hosted Kimi K2.5
    }

    # Models that use the Responses API with explicit reasoning budget.
    REASONING_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
        "gpt-5.4": {
            "effort": "low",
            "max_output_tokens": 2048,
        },
        "o3": {
            "effort": "medium",
            "max_output_tokens": 2048,
        },
    }
    GEMINI_MIN_CHAT_MAX_TOKENS = 2048

    @classmethod
    def resolve_model_alias(cls, model: str) -> str:
        """Resolve a model alias to the full OpenAI model ID."""
        return cls.MODEL_ALIASES.get(model.lower(), model)

    @staticmethod
    def _is_fireworks_model(model: str) -> bool:
        normalized = (model or "").strip().lower()
        return normalized.startswith("accounts/fireworks/models/")

    @staticmethod
    def _is_gemini_model(model: str) -> bool:
        normalized = (model or "").strip().lower()
        return normalized.startswith("gemini-")

    @staticmethod
    def _get_model_api_style(model: str) -> str:
        """Determine API parameter style for a model.

        Returns one of:
            'openai_new' — gpt-5.x: uses max_completion_tokens, no stop
            'fireworks'  — Fireworks-hosted: skip max_tokens entirely (reasoning models)
            'openai'     — everything else: standard max_tokens + stop
        """
        normalized = (model or "").strip().lower()
        if normalized.startswith("accounts/fireworks/models/"):
            return "fireworks"
        if normalized.startswith("gemini-"):
            return "openai"
        if "gpt-5" in normalized or normalized in ("o3", "o3-mini", "o4-mini"):
            return "openai_new"
        return "openai"

    @classmethod
    def _get_reasoning_model_config(cls, model: str) -> Optional[Dict[str, Any]]:
        normalized = (model or "").strip().lower()
        return cls.REASONING_MODEL_CONFIG.get(normalized)

    @staticmethod
    def _to_response_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert chat-style messages to Responses API input items."""
        input_items: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                continue

            raw_content = message.get("content", "")
            if isinstance(raw_content, list):
                content: List[Dict[str, Any]] = []
                for part in raw_content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type == "text":
                        content.append({"type": "input_text", "text": part.get("text", "")})
                    elif part_type == "image_url":
                        image = part.get("image_url") or {}
                        content.append(
                            {
                                "type": "input_image",
                                "image_url": image.get("url"),
                                "detail": image.get("detail", "auto"),
                            }
                        )
            else:
                content = [{"type": "input_text", "text": str(raw_content)}]

            input_items.append({"role": role, "content": content})

        return input_items

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text

        chunks: List[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif hasattr(usage, "dict"):
            usage = usage.dict()
        if not isinstance(usage, dict):
            return {}

        prompt_details = usage.get("prompt_tokens_details") or {}
        input_details = usage.get("input_tokens_details") or {}
        return {
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "cached_input_tokens": prompt_details.get("cached_tokens", input_details.get("cached_tokens", 0)),
        }

    def __init__(self, conf: DictConfig):
        """
        Initialize the chat model.
        :param conf: the configuration of the language model
        """
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        model_name = self.resolve_model_alias(self.generation_params.model)
        if self._is_fireworks_model(model_name):
            api_key = (os.getenv("FIREWORKS_API_KEY") or "").strip()
            if not api_key:
                raise ValueError("No FIREWORKS_API_KEY provided")
            base_url = (
                os.getenv("FIREWORKS_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.fireworks.ai/inference/v1"
            ).strip()
        elif self._is_gemini_model(model_name):
            api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
            if not api_key:
                raise ValueError("No GEMINI_API_KEY provided")
            base_url = (
                os.getenv("GEMINI_BASE_URL")
                or "https://generativelanguage.googleapis.com/v1beta/openai/"
            ).strip()
        else:
            api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
            if not api_key:
                raise ValueError("No OPENAI_API_KEY provided")
            base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._validate_conf()
        self.verbose = self.llm_conf.verbose
        self.verbose = True
        self.message_history: List[Dict] = []
        self.keep_message_history = self.llm_conf.keep_message_history

    def _validate_conf(self):
        if self.generation_params.stream:
            raise ValueError("Streaming not supported")

    # @retry(Timeout, tries=3)
    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
        request_timeout: int = 40,
    ):
        """
        Generate a response autoregressively.
        :param prompt: A string with the input to the language model.
        :param image: Image input
        :param stop: A string that determines when to stop generation
        :param max_length: The max number of tokens to generate.
        :param request_timeout: maximum time before timeout.
        :param generation_args: contains arguments like the grammar definition. We don't use this here
        """

        params = OmegaConf.to_object(self.generation_params)

        # Resolve model alias
        params["model"] = self.resolve_model_alias(params["model"])

        # Override stop if provided
        if stop is None and len(self.generation_params.stop) > 0:
            stop = self.generation_params.stop
        params["stop"] = stop

        # Override max_length if provided
        if max_length is not None:
            params["max_tokens"] = max_length

        messages = self.message_history.copy()
        # Add system message if no messages
        if len(messages) == 0:
            messages.append({"role": "system", "content": self.llm_conf.system_message})

        params["request_timeout"] = request_timeout
        if type(prompt) is str:
            # Add current message
            messages.append({"role": "user", "content": prompt})

        else:
            # Multimodal prompt
            image_detail = "low"  # high/low/auto
            messages.append(generate_message(prompt, image_detail=image_detail))

        temperature = params.get("temperature", 0.7)
        reasoning_cfg = self._get_reasoning_model_config(params["model"])
        api_style = self._get_model_api_style(params["model"])
        token_limit = params.get("max_tokens")
        if self._is_gemini_model(params["model"]) and token_limit is not None:
            # Gemini reasoning models often spend a large share of the output budget
            # before emitting the first visible action line. Keep a higher floor here
            # so planner prompts can reach `Agent_{id}_Action: ...` reliably.
            token_limit = max(token_limit, self.GEMINI_MIN_CHAT_MAX_TOKENS)

        if reasoning_cfg:
            # Responses API path (gpt-5.4 with reasoning)
            response_kwargs: Dict[str, Any] = {
                "model": params["model"],
                "input": self._to_response_input(messages),
                "reasoning": {"effort": reasoning_cfg["effort"]},
                "max_output_tokens": (
                    max_length if max_length is not None else reasoning_cfg["max_output_tokens"]
                ),
                "timeout": request_timeout,
            }
            if self.llm_conf.system_message:
                response_kwargs["instructions"] = self.llm_conf.system_message
            response = self.client.responses.create(**response_kwargs)
            text_response = self._extract_response_text(response)
            maybe_append_usage_event(
                provider="openai",
                model=params["model"],
                usage=self._extract_usage(response),
                source="habitat_llm.openai_chat.responses",
            )
        else:
            # Chat Completions API path
            completion_kwargs: Dict[str, Any] = {
                "model": params["model"],
                "messages": messages,
                "timeout": request_timeout,
            }

            # Token limit + stop: depends on provider/model family
            if api_style == "fireworks":
                pass  # Fireworks reasoning models need unrestricted output
            elif api_style == "openai_new":
                # gpt-5.x: max_completion_tokens, no stop
                if token_limit is not None:
                    completion_kwargs["max_completion_tokens"] = token_limit
            else:
                # Standard OpenAI / other providers
                if token_limit is not None:
                    completion_kwargs["max_tokens"] = token_limit
                if stop is not None:
                    completion_kwargs["stop"] = stop

            # Temperature: all chat-completion models support it
            completion_kwargs["temperature"] = temperature

            response = self.client.chat.completions.create(**completion_kwargs)
            text_response = response.choices[0].message.content
            maybe_append_usage_event(
                provider="openai",
                model=params["model"],
                usage=self._extract_usage(response),
                source="habitat_llm.openai_chat.chat_completions",
            )
        self.response = text_response

        # Update message history
        if self.keep_message_history:
            self.message_history = messages.copy()
            self.message_history.append({"role": "assistant", "content": text_response})

        if stop is not None:
            text_response = text_response.split(stop)[0]
        return text_response

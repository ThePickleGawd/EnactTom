#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib import error, request

from omegaconf import DictConfig

from enacttom.api_costs import maybe_append_usage_event
from habitat_llm.llm.base_llm import BaseLLM, LLMRequestError, Prompt

# Load .env file if it exists (for API keys)
_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_env_file)
    except ImportError:
        # Fallback: manually parse .env if python-dotenv is not installed.
        with open(_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(
                        key.strip(), value.strip().strip('"').strip("'")
                    )


class AnthropicClaude(BaseLLM):
    """
    LLM implementation using Anthropic's Messages API.
    Uses environment variable: ANTHROPIC_API_KEY

    Supports model aliases:
        sonnet, sonnet-4.6            -> claude-sonnet-4-6
        haiku, haiku-4.5              -> claude-haiku-4-5-20251001
        opus, opus-4.6                -> claude-opus-4-6
        sonnet-4.5                    -> claude-sonnet-4-5-20250929
        opus-4.5                      -> claude-opus-4-5-20251101
    """

    MODEL_ALIASES: Dict[str, str] = {
        # Anthropic native model names — default aliases point to latest
        "sonnet": "claude-sonnet-4-6",
        "sonnet-4.6": "claude-sonnet-4-6",
        "sonnet4.6": "claude-sonnet-4-6",
        "sonnet-4.5": "claude-sonnet-4-5-20250929",
        "sonnet4.5": "claude-sonnet-4-5-20250929",
        "haiku": "claude-haiku-4-5-20251001",
        "haiku-4.5": "claude-haiku-4-5-20251001",
        "haiku4.5": "claude-haiku-4-5-20251001",
        "opus": "claude-opus-4-6",
        "opus-4.6": "claude-opus-4-6",
        "opus4.6": "claude-opus-4-6",
        "opus-4.5": "claude-opus-4-5-20251101",
        "opus4.5": "claude-opus-4-5-20251101",
        # Bedrock IDs mapped for convenience if reused with this provider.
        "us.anthropic.claude-sonnet-4-6-v1:0": "claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "claude-sonnet-4-5-20250929",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": "claude-haiku-4-5-20251001",
        "us.anthropic.claude-opus-4-6-v1:0": "claude-opus-4-6",
        "us.anthropic.claude-opus-4-5-20251101-v1:0": "claude-opus-4-5-20251101",
    }

    @classmethod
    def resolve_model_alias(cls, model: str) -> str:
        """Resolve a model alias to a full Anthropic model ID."""
        return cls.MODEL_ALIASES.get(model.lower(), model)

    def __init__(self, conf: DictConfig):
        """
        Initialize the Anthropic model.
        :param conf: the configuration of the language model
        """
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params

        self.api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        if not self.api_key:
            raise ValueError("No ANTHROPIC_API_KEY provided")

        self.api_base = getattr(self.llm_conf, "api_base", "https://api.anthropic.com")
        self.verbose = getattr(self.llm_conf, "verbose", True)
        self.message_history: List[Dict] = []
        self.keep_message_history = getattr(
            self.llm_conf, "keep_message_history", False
        )
        self.system_message = getattr(
            self.llm_conf, "system_message", "You are an expert at task planning."
        )

    def _build_multimodal_content(self, prompt: Prompt) -> List[Dict]:
        """Convert prompt tuples into Anthropic content blocks."""
        content: List[Dict] = []
        for prompt_type, prompt_value in prompt:
            if prompt_type == "text":
                content.append({"type": "text", "text": prompt_value})
            elif prompt_value.startswith("data:"):
                # Expecting format: data:<media_type>;base64,<base64_data>
                parts = prompt_value.split(",", 1)
                if len(parts) != 2:
                    continue
                media_type = parts[0].split(";")[0].replace("data:", "")
                base64_data = parts[1]
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data,
                        },
                    }
                )
        return content

    def _build_stop_sequences(self, stop: Optional[str]) -> Optional[List[str]]:
        """Normalize stop into list[str] for Anthropic API."""
        if not stop:
            return None
        if isinstance(stop, str):
            return [stop]
        return [str(s) for s in stop if str(s)]

    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
        request_timeout: int = 40,
    ):
        """
        Generate a response using Anthropic's Messages API.
        :param prompt: input prompt string or multimodal tuples
        :param stop: stop sequence(s)
        :param max_length: max tokens to generate
        :param generation_args: unused
        :param request_timeout: request timeout in seconds
        """
        del generation_args  # Not used by this client.

        model = self.resolve_model_alias(self.generation_params.model)

        if (
            stop is None
            and hasattr(self.generation_params, "stop")
            and self.generation_params.stop
        ):
            stop = self.generation_params.stop
        stop_sequences = self._build_stop_sequences(stop)

        max_tokens = (
            max_length
            if max_length is not None
            else getattr(self.generation_params, "max_tokens", 1024)
        )

        messages = self.message_history.copy()
        if isinstance(prompt, str):
            user_content = [{"type": "text", "text": prompt}]
        else:
            user_content = self._build_multimodal_content(prompt)
        messages.append({"role": "user", "content": user_content})

        request_body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if self.system_message:
            request_body["system"] = self.system_message

        # Claude models expect either temperature or top_p.
        if hasattr(self.generation_params, "temperature"):
            request_body["temperature"] = self.generation_params.temperature
        elif hasattr(self.generation_params, "top_p"):
            request_body["top_p"] = self.generation_params.top_p

        if stop_sequences:
            request_body["stop_sequences"] = stop_sequences

        payload = json.dumps(request_body).encode("utf-8")
        req = request.Request(
            url=f"{self.api_base.rstrip('/')}/v1/messages",
            data=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=request_timeout) as resp:
                response_body = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            details = ""
            try:
                details = e.read().decode("utf-8")
            except Exception:
                details = str(e)
            raise LLMRequestError(
                f"Anthropic API HTTP {e.code}: {details}",
                status_code=e.code,
                headers=dict(e.headers.items()) if e.headers else None,
                retryable=e.code in {408, 409, 429, 500, 502, 503, 504},
            ) from e
        except Exception as e:
            raise LLMRequestError(
                f"Anthropic API request failed: {e}",
                retryable=isinstance(e, (TimeoutError, error.URLError, ConnectionError, OSError)),
            ) from e

        content_blocks = response_body.get("content", [])
        text_response = "".join(
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        ).strip()
        usage = response_body.get("usage") or {}
        maybe_append_usage_event(
            provider="anthropic",
            model=model,
            usage={
                "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cached_input_tokens": usage.get("cache_read_input_tokens", 0),
            },
            source="habitat_llm.anthropic_claude.messages",
        )
        self.response = text_response

        if self.keep_message_history:
            self.message_history = messages.copy()
            self.message_history.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text_response}],
                }
            )

        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq and stop_seq in text_response:
                    text_response = text_response.split(stop_seq)[0]
                    break

        return text_response

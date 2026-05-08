#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from omegaconf import DictConfig

from enacttom.api_costs import maybe_append_usage_event
from habitat_llm.llm.base_llm import BaseLLM, LLMRequestError, Prompt

# Load .env file if it exists (for AWS credentials)
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


class BedrockClaude(BaseLLM):
    """
    LLM implementation using AWS Bedrock with multiple model families.
    Uses environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

    Supports model aliases for convenience:

    Claude (Anthropic):
        sonnet, sonnet-4.6       -> us.anthropic.claude-sonnet-4-6-v1:0
        haiku, haiku-4.5         -> us.anthropic.claude-haiku-4-5-20251001-v1:0
        opus, opus-4.6           -> us.anthropic.claude-opus-4-6-v1:0
        sonnet-4.5               -> us.anthropic.claude-sonnet-4-5-20250929-v1:0
        opus-4.5                 -> us.anthropic.claude-opus-4-5-20251101-v1:0

    Qwen (Alibaba):
        qwen3-80b, qwen3-next    -> qwen.qwen3-next-80b-a3b
        qwen3-vl, qwen3-vl-235b  -> qwen.qwen3-vl-235b-a22b

    Kimi (Moonshot):
        kimi-k2, kimi-thinking   -> moonshot.kimi-k2-thinking

    Mistral:
        ministral-8b             -> mistral.ministral-3-8b-instruct
        ministral-14b            -> mistral.ministral-3-14b-instruct
        mistral-large            -> mistral.mistral-large-3-675b-instruct
    """

    # Model aliases mapping short names to full Bedrock model IDs
    MODEL_ALIASES: Dict[str, str] = {
        # ============ Claude (Anthropic) ============
        # Claude Sonnet 4.6 (latest)
        "sonnet": "us.anthropic.claude-sonnet-4-6-v1:0",
        "sonnet-4.6": "us.anthropic.claude-sonnet-4-6-v1:0",
        "sonnet4.6": "us.anthropic.claude-sonnet-4-6-v1:0",
        # Claude Sonnet 4.5
        "sonnet-4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "sonnet4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        # Claude Haiku 4.5
        "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "haiku4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        # Claude Opus 4.6 (latest)
        "opus": "us.anthropic.claude-opus-4-6-v1:0",
        "opus-4.6": "us.anthropic.claude-opus-4-6-v1:0",
        "opus4.6": "us.anthropic.claude-opus-4-6-v1:0",
        # Claude Opus 4.5
        "opus-4.5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "opus4.5": "us.anthropic.claude-opus-4-5-20251101-v1:0",

        # ============ Qwen (Alibaba) ============
        # Qwen3 Next 80B A3B Instruct
        "qwen3-80b": "qwen.qwen3-next-80b-a3b",
        "qwen3-next": "qwen.qwen3-next-80b-a3b",
        "qwen3-next-80b": "qwen.qwen3-next-80b-a3b",
        # Qwen3 VL 235B A22B (Vision-Language)
        "qwen3-vl": "qwen.qwen3-vl-235b-a22b",
        "qwen3-vl-235b": "qwen.qwen3-vl-235b-a22b",

        # ============ Kimi (Moonshot) ============
        # Kimi K2 Thinking
        "kimi-k2": "moonshot.kimi-k2-thinking",
        "kimi-thinking": "moonshot.kimi-k2-thinking",
        "kimi-k2-thinking": "moonshot.kimi-k2-thinking",

        # ============ Mistral ============
        # Ministral 3 8B Instruct
        "ministral-8b": "mistral.ministral-3-8b-instruct",
        "ministral-3-8b": "mistral.ministral-3-8b-instruct",
        # Ministral 3 14B Instruct
        "ministral-14b": "mistral.ministral-3-14b-instruct",
        "ministral-3-14b": "mistral.ministral-3-14b-instruct",
        # Mistral Large 3 (675B)
        "mistral-large": "mistral.mistral-large-3-675b-instruct",
        "mistral-large-3": "mistral.mistral-large-3-675b-instruct",
    }

    # Models that use the Converse API (non-Claude models)
    CONVERSE_API_MODELS = {
        "qwen.", "moonshot.", "mistral."
    }

    @classmethod
    def resolve_model_alias(cls, model: str) -> str:
        """Resolve a model alias to the full Bedrock model ID."""
        return cls.MODEL_ALIASES.get(model.lower(), model)

    @classmethod
    def uses_converse_api(cls, model_id: str) -> bool:
        """Check if a model should use the Converse API instead of invoke_model."""
        return any(model_id.startswith(prefix) for prefix in cls.CONVERSE_API_MODELS)

    def __init__(self, conf: DictConfig):
        """
        Initialize the Bedrock model.
        :param conf: the configuration of the language model
        """
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params

        # Initialize boto3 client - uses env vars automatically
        region = os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=region)

        self.verbose = getattr(self.llm_conf, "verbose", True)
        self.message_history: List[Dict] = []
        self.keep_message_history = getattr(self.llm_conf, "keep_message_history", False)
        self.system_message = getattr(self.llm_conf, "system_message", "You are an expert at task planning.")

    def _raise_client_error(self, exc: ClientError) -> None:
        err = exc.response.get("Error", {})
        metadata = exc.response.get("ResponseMetadata", {})
        code = err.get("Code", "")
        message = err.get("Message", str(exc))
        status_code = metadata.get("HTTPStatusCode")
        headers = metadata.get("HTTPHeaders") or {}
        retryable_codes = {
            "ThrottlingException",
            "TooManyRequestsException",
            "RequestTimeout",
            "ServiceUnavailableException",
            "InternalServerException",
        }
        retryable = status_code in {408, 409, 429, 500, 502, 503, 504} or code in retryable_codes
        raise LLMRequestError(
            f"Bedrock {code or 'ClientError'}: {message}",
            status_code=status_code,
            headers=headers,
            retryable=retryable,
        ) from exc

    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
        request_timeout: int = 40,
    ):
        """
        Generate a response using AWS Bedrock.
        :param prompt: A string with the input to the language model.
        :param stop: A string that determines when to stop generation
        :param max_length: The max number of tokens to generate.
        :param request_timeout: maximum time before timeout (not used directly by boto3)
        :param generation_args: contains arguments like the grammar definition. Not used here.
        """
        # Get model ID (resolve alias if provided)
        model_id = self.resolve_model_alias(self.generation_params.model)

        # Override stop if provided
        if stop is None and hasattr(self.generation_params, "stop") and self.generation_params.stop:
            stop = self.generation_params.stop

        # Override max_length if provided
        max_tokens = max_length if max_length is not None else self.generation_params.max_tokens

        # Route to appropriate API based on model
        if self.uses_converse_api(model_id):
            return self._generate_converse(prompt, model_id, max_tokens, stop)
        else:
            return self._generate_claude(prompt, model_id, max_tokens, stop)

    def _generate_claude(
        self,
        prompt: Prompt,
        model_id: str,
        max_tokens: int,
        stop: Optional[str],
    ) -> str:
        """Generate using Claude's native API format."""
        # Build messages
        messages = self.message_history.copy()

        # Add current message
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            # Multimodal prompt - convert to Claude format
            content = []
            for prompt_type, prompt_value in prompt:
                if prompt_type == "text":
                    content.append({"type": "text", "text": prompt_value})
                else:
                    # Image - extract base64 data
                    # Expecting format: "data:image/jpeg;base64,<base64_data>"
                    if prompt_value.startswith("data:"):
                        parts = prompt_value.split(",", 1)
                        if len(parts) == 2:
                            media_type = parts[0].split(";")[0].replace("data:", "")
                            base64_data = parts[1]
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data,
                                }
                            })
            messages.append({"role": "user", "content": content})

        # Build request body for Claude
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": messages,
        }

        # Add system message
        if self.system_message:
            request_body["system"] = self.system_message

        # Add optional parameters
        # Note: Claude Sonnet 4.5 doesn't allow both temperature and top_p
        if hasattr(self.generation_params, "temperature"):
            request_body["temperature"] = self.generation_params.temperature
        elif hasattr(self.generation_params, "top_p"):
            request_body["top_p"] = self.generation_params.top_p
        if stop:
            request_body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        # Call Bedrock
        try:
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
        except ClientError as exc:
            self._raise_client_error(exc)

        # Parse response
        response_body = json.loads(response["body"].read())
        text_response = response_body["content"][0]["text"]
        usage = response_body.get("usage") or {}
        if isinstance(usage, dict) and any(
            usage.get(key)
            for key in (
                "input_tokens",
                "inputTokens",
                "output_tokens",
                "outputTokens",
                "cache_read_input_tokens",
                "cacheReadInputTokens",
            )
        ):
            maybe_append_usage_event(
                provider="bedrock",
                model=model_id,
                usage={
                    "input_tokens": usage.get("input_tokens", usage.get("inputTokens", 0)),
                    "output_tokens": usage.get("output_tokens", usage.get("outputTokens", 0)),
                    "cached_input_tokens": usage.get(
                        "cache_read_input_tokens",
                        usage.get("cacheReadInputTokens", 0),
                    ),
                },
                source="habitat_llm.bedrock_claude.invoke_model",
            )
        self.response = text_response

        # Update message history
        if self.keep_message_history:
            self.message_history = messages.copy()
            self.message_history.append({"role": "assistant", "content": text_response})

        # Handle stop sequence
        if stop is not None and stop in text_response:
            text_response = text_response.split(stop)[0]

        return text_response

    def _generate_converse(
        self,
        prompt: Prompt,
        model_id: str,
        max_tokens: int,
        stop: Optional[str],
    ) -> str:
        """
        Generate using Bedrock's Converse API.
        Works with Qwen, Mistral, Kimi, and other non-Claude models.
        """
        # Build messages in Converse API format
        messages = []

        # Add message history
        for msg in self.message_history:
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    messages.append({
                        "role": "user",
                        "content": [{"text": msg["content"]}]
                    })
                else:
                    messages.append({"role": "user", "content": msg["content"]})
            else:
                messages.append({
                    "role": "assistant",
                    "content": [{"text": msg["content"]}]
                })

        # Add current message
        if isinstance(prompt, str):
            messages.append({
                "role": "user",
                "content": [{"text": prompt}]
            })
        else:
            # Multimodal prompt - convert to Converse format
            content = []
            for prompt_type, prompt_value in prompt:
                if prompt_type == "text":
                    content.append({"text": prompt_value})
                else:
                    # Image - extract base64 data
                    if prompt_value.startswith("data:"):
                        parts = prompt_value.split(",", 1)
                        if len(parts) == 2:
                            media_type = parts[0].split(";")[0].replace("data:", "")
                            base64_data = parts[1]
                            # Map media type to format
                            format_map = {
                                "image/jpeg": "jpeg",
                                "image/png": "png",
                                "image/gif": "gif",
                                "image/webp": "webp",
                            }
                            img_format = format_map.get(media_type, "jpeg")
                            content.append({
                                "image": {
                                    "format": img_format,
                                    "source": {
                                        "bytes": base64_data
                                    }
                                }
                            })
            messages.append({"role": "user", "content": content})

        # Build inference config
        inference_config = {
            "maxTokens": max_tokens,
        }

        # Add temperature if specified
        if hasattr(self.generation_params, "temperature"):
            inference_config["temperature"] = self.generation_params.temperature
        elif hasattr(self.generation_params, "top_p"):
            inference_config["topP"] = self.generation_params.top_p

        # Add stop sequences if specified
        if stop:
            inference_config["stopSequences"] = [stop] if isinstance(stop, str) else stop

        # Build request kwargs
        request_kwargs = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        # Add system message if specified
        if self.system_message:
            request_kwargs["system"] = [{"text": self.system_message}]

        # Call Bedrock Converse API.
        # Some models (notably certain Mistral variants) reject stopSequences.
        try:
            response = self.client.converse(**request_kwargs)
        except ClientError as e:
            err = e.response.get("Error", {})
            code = err.get("Code", "")
            message = err.get("Message", "")
            normalized_message = message.lower().replace(" ", "")
            should_retry_without_stop = (
                "stopSequences" in inference_config
                and code == "ValidationException"
                and "stopsequences" in normalized_message
            )
            if not should_retry_without_stop:
                self._raise_client_error(e)

            inference_config.pop("stopSequences", None)
            try:
                response = self.client.converse(**request_kwargs)
            except ClientError as exc:
                self._raise_client_error(exc)

        # Parse response
        text_response = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage") or {}
        if isinstance(usage, dict) and any(
            usage.get(key)
            for key in (
                "inputTokens",
                "input_tokens",
                "outputTokens",
                "output_tokens",
                "cacheReadInputTokens",
                "cache_read_input_tokens",
            )
        ):
            maybe_append_usage_event(
                provider="bedrock",
                model=model_id,
                usage={
                    "input_tokens": usage.get("inputTokens", usage.get("input_tokens", 0)),
                    "output_tokens": usage.get("outputTokens", usage.get("output_tokens", 0)),
                    "cached_input_tokens": usage.get(
                        "cacheReadInputTokens",
                        usage.get("cache_read_input_tokens", 0),
                    ),
                },
                source="habitat_llm.bedrock_claude.converse",
            )
        self.response = text_response

        # Update message history (convert back to simple format)
        if self.keep_message_history:
            self.message_history.append({"role": "user", "content": prompt if isinstance(prompt, str) else prompt})
            self.message_history.append({"role": "assistant", "content": text_response})

        # Handle stop sequence
        if stop is not None and stop in text_response:
            text_response = text_response.split(stop)[0]

        return text_response

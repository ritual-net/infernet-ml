"""
Library containing functions for accessing closed source models.

Currently, 3 APIs are supported: OPENAI, PERPLEXITYAI, and GOOSEAI.

Depending on the API being used, the appropriate API key must be specified

as an environment variable as "{provider}_API_KEY".

"""
import os
from typing import Any, cast

import requests

from infernet_ml.utils.service_models import (
    CSSCompletionParams,
    CSSEmbeddingParams,
    CSSRequest,
)
from infernet_ml.workflows.exceptions import RetryableException, ServiceException


def open_ai_helper(req: CSSRequest) -> tuple[str, dict[str, Any]]:
    """
    Returns base url, processed input.
    """
    match req:
        case CSSRequest(model=model_name, params=CSSCompletionParams(messages=msgs)):
            return "https://api.openai.com/v1/", {
                "model": model_name,
                "messages": [msg.model_dump() for msg in msgs],
            }

        case CSSRequest(model=model_name, params=CSSEmbeddingParams(input=input)):
            return "https://api.openai.com/v1/", {
                "model": model_name,
                "input": input,
            }
        case _:
            raise ServiceException(f"Unsupported request {req}")


def ppl_ai_helper(req: CSSRequest) -> tuple[str, dict[str, Any]]:
    """
    Returns base url, processed input.
    """
    match req:
        case CSSRequest(model=model_name, params=CSSCompletionParams(messages=msgs)):
            return "https://api.perplexity.ai/", {
                "model": model_name,
                "messages": [msg.model_dump() for msg in msgs],
            }
        case _:
            raise ServiceException(f"Unsupported request {req}")


def goose_ai_helper(req: CSSRequest) -> tuple[str, dict[str, Any]]:
    """
    Returns base url, processed input.
    """
    match req:
        case CSSRequest(model=model_name, params=CSSCompletionParams(messages=msgs)):
            if len(msgs) != 1:
                raise ServiceException(
                    "GOOSE AI API only accepts one message from role user!"
                )
            inp = msgs[0].content
            return f"https://api.goose.ai/v1/engines/{model_name}/", {"prompt": inp}
        case _:
            raise ServiceException(f"Unsupported request {req}")


PROVIDERS: dict[str, Any] = {
    "OPENAI": {
        "input_func": open_ai_helper,
        "endpoints": {
            "completions": {
                "real_endpoint": "chat/completions",
                "proc": lambda result: result["choices"][0]["message"]["content"],
            },
            "embeddings": {
                "real_endpoint": "embeddings",
                "proc": lambda result: result["data"][0]["embedding"],
            },
        },
    },
    "PERPLEXITYAI": {
        "input_func": ppl_ai_helper,
        "endpoints": {
            "completions": {
                "real_endpoint": "chat/completions",
                "proc": lambda result: result["choices"][0]["message"]["content"],
            }
        },
    },
    "GOOSEAI": {
        "input_func": goose_ai_helper,
        "endpoints": {
            "completions": {
                "real_endpoint": "completions",
                "proc": lambda result: result["choices"][0]["text"],
            }
        },
    },
}


def validate(provider: str, endpoint: str) -> None:
    """helper function to validate provider and endpoint

    Args:
        provider (str): provider used
        endpoint (str): end point used

    Raises:
        ServiceException: if API Key not specified or an unsupported
        provider or endpoint specified.
    """
    if (api_key := f"{provider}_API_KEY") not in os.environ:
        raise ServiceException(f"Environment variable {api_key} not found!")

    if provider not in PROVIDERS:
        raise ServiceException("Provider not supported!")

    if endpoint not in PROVIDERS[provider]["endpoints"]:
        raise ServiceException("Endpoint not supported for your provider!")


def css_mux(provider: str, req: CSSRequest) -> str:
    """
    Args:
        provider: Closed AI provider
        endpoint: options: ("completions", "embeddings")
        req: CSSRequest
    Returns:
        response: processed output from api
    """

    if (api_key := f"{provider}_API_KEY") not in os.environ:
        raise ServiceException(f"Environment variable {api_key} not found!")

    if provider not in PROVIDERS:
        raise ServiceException("Provider not supported!")

    if req.params.endpoint not in PROVIDERS[provider]["endpoints"]:
        raise ServiceException("Endpoint not supported for your provider!")

    real_endpoint = PROVIDERS[provider]["endpoints"][req.params.endpoint][
        "real_endpoint"
    ]
    base_url, proc_input = PROVIDERS[provider]["input_func"](req)
    url = f"{base_url}{real_endpoint}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ[api_key]}",
    }
    result = requests.post(url, headers=headers, json=proc_input)

    if result.status_code != 200:
        match provider:
            case "OPENAI" | "GOOSEAI":
                # https://help.openai.com/en/articles/6891839-api-error-code-guidance
                if result.status_code == 429 or result.status_code == 500:
                    raise RetryableException(result.text)
            case "PERPLEXITYAI":
                if result.status_code == 429:
                    raise RetryableException(result.text)
            case _:
                raise ServiceException(result.text)

    post_proc = PROVIDERS[provider]["endpoints"][req.params.endpoint]["proc"]
    return cast(str, post_proc(result.json()))

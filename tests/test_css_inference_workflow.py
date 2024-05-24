"""
simple test for a CSS Inference Workflow
"""
import os

import pytest

import json

from infernet_ml.utils.service_models import (
    ConvoMessage,
    CSSCompletionParams,
    CSSRequest,
)
from infernet_ml.workflows.inference.css_inference_workflow import CSSInferenceWorkflow


@pytest.mark.parametrize(
    "provider, endpoint",
    [
        ("OPENAI", "completions"),
        ("OPENAI", "embeddings"),
        ("PERPLEXITYAI", "completions"),
        ("GOOSEAI", "completions"),
        ("CHASMNET", "prompt"),
        ("CHASMNET", "workflows"),
    ],
)
def test_init(provider: str, endpoint: str) -> None:
    _environ = dict(os.environ)  # or os.environ.copy()
    try:
        os.environ[f"{provider}_API_KEY"] = "test_key"
        if provider != "CHASMNET":
            _: CSSInferenceWorkflow = CSSInferenceWorkflow(provider, endpoint)
        else:
            _: 
    finally:
        os.environ.clear()
        os.environ.update(_environ)


@pytest.mark.parametrize(
    "provider, model, messages",
    [
        (
            "OPENAI",
            "gpt-3.5-turbo-16k",
            [ConvoMessage(role="user", content="hi how are you")],
        ),
        (
            "PERPLEXITYAI",
            "mistral-7b-instruct",
            [ConvoMessage(role="user", content="hi how are you")],
        ),
        ("GOOSEAI", "gpt-j-6b", [ConvoMessage(role="user", content="hi how are you")]),
    ],
)
def test_completion_inferences(
    provider: str, model: str, messages: list[ConvoMessage]
) -> None:
    if os.environ.get(f"{provider}_API_KEY"):
        params: CSSCompletionParams = CSSCompletionParams(
            endpoint="completions", messages=messages
        )
        req: CSSRequest = CSSRequest(model=model, params=params)
        workflow: CSSInferenceWorkflow = CSSInferenceWorkflow(provider, "completions")
        workflow.setup()
        res = workflow.inference(req.model_dump())
        assert len(res)


@pytest.mark.parametrize(
    "provider, endpoint_id, body",
    [
        (
            "CHASMNET",
            "111",
            json.loads('{"input": {"test_input_1": "apple"}}'),
        ),
    ],
)
def test_chasm_prompt_inferences(
    provider: str, model: str, messages: list[ConvoMessage]
) -> None:
    if os.environ.get(f"{provider}_API_KEY"):
        params: ChasmPromptParams = ChasmPromptParams(
            endpoint="prompt", messages=messages
        )
        req: ChasmRequest = ChasmRequest(model=model, params=params)
        workflow: CSSInferenceWorkflow = CSSInferenceWorkflow(provider, "prompt")
        workflow.setup()
        res = workflow.inference(req.model_dump())
        assert len(res)


@pytest.mark.parametrize(
    "provider, endpoint_id, body",
    [
        (
            "CHASMNET",
            "111",
            json.loads('{"input": {"test_input_1": "apple"}}'),
        ),
    ],
)
def test_chasm_workflows_inferences(
    provider: str, model: str, messages: list[ConvoMessage]
) -> None:
    if os.environ.get(f"{provider}_API_KEY"):
        params: ChasmPromptParams = ChasmPromptParams(
            endpoint="workflows", messages=messages
        )
        req: ChasmRequest = ChasmRequest(model=model, params=params)
        workflow: CSSInferenceWorkflow = CSSInferenceWorkflow(provider, "workflows")
        workflow.setup()
        res = workflow.inference(req.model_dump())
        assert len(res)
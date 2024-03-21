import os

import pytest

from infernet_ml.utils.hf_types import (
    HFClassificationInferenceInput,
    HFSummarizationConfig,
    HFSummarizationInferenceInput,
    HFTextGenerationInferenceInput,
)
from infernet_ml.workflows.inference.hf_inference_client_workflow import (
    HFInferenceClientWorkflow,
)

token = os.environ.get("HF_TOKEN")


@pytest.fixture
def text_classification_workflow() -> HFInferenceClientWorkflow:
    return HFInferenceClientWorkflow(
        task="text_classification",
        token=token,
    )


@pytest.fixture
def token_classification_workflow() -> HFInferenceClientWorkflow:
    return HFInferenceClientWorkflow(
        task="token_classification",
        token=token,
    )


@pytest.fixture
def summarization_workflow() -> HFInferenceClientWorkflow:
    return HFInferenceClientWorkflow(
        task="summarization",
        token=token,
    )


@pytest.fixture
def text_generation_workflow() -> HFInferenceClientWorkflow:
    return HFInferenceClientWorkflow(
        task="text_generation",
        token=token,
    )


def test_do_setup_text_classification(
    text_classification_workflow: HFInferenceClientWorkflow,
) -> None:
    text_classification_workflow.setup()
    assert text_classification_workflow.client is not None
    assert text_classification_workflow.task_argspec is not None


def test_do_preprocessing_text_classification(
    text_classification_workflow: HFInferenceClientWorkflow,
) -> None:
    input_data = {"text": "Hello, world!"}
    preprocessed_data = text_classification_workflow.do_preprocessing(input_data)
    assert preprocessed_data.model_dump()["text"] == input_data["text"]


def test_do_run_model_text_classification(
    text_classification_workflow: HFInferenceClientWorkflow,
) -> None:
    text_classification_workflow.setup()
    input_data = HFClassificationInferenceInput(
        text="Decentralizing AI using crypto is awesome!"
    )
    output_data = text_classification_workflow.do_run_model(input_data)
    assert output_data.get("output")[0].get("label") == "POSITIVE"  # type: ignore
    assert output_data.get("output")[0].get("score") > 0.6  # type: ignore


def test_do_postprocessing_text_classification(
    text_classification_workflow: HFInferenceClientWorkflow,
) -> None:
    output_data = {"prediction": "positive"}
    postprocessed_data = text_classification_workflow.do_postprocessing(
        output_data, output_data
    )
    assert postprocessed_data == output_data


def test_do_setup_token_classification(
    token_classification_workflow: HFInferenceClientWorkflow,
) -> None:
    token_classification_workflow.setup()
    assert token_classification_workflow.client is not None
    assert token_classification_workflow.task_argspec is not None


def test_do_preprocessing_token_classification(
    token_classification_workflow: HFInferenceClientWorkflow,
) -> None:
    input_data = {"text": "Hello, world!"}
    preprocessed_data = token_classification_workflow.do_preprocessing(input_data)
    assert preprocessed_data.model_dump()["text"] == input_data["text"]


def test_do_run_model_token_classification(
    token_classification_workflow: HFInferenceClientWorkflow,
) -> None:
    token_classification_workflow.setup()
    input_data = HFClassificationInferenceInput(
        text="Ritual makes AI x crypto a great combination!"
    )
    output_data = token_classification_workflow.do_run_model(input_data)
    assert output_data.get("output")[0].get("entity_group") == "MISC"  # type: ignore
    assert output_data.get("output")[0].get("score") > 0.8  # type: ignore


def test_do_postprocessing_token_classification(
    token_classification_workflow: HFInferenceClientWorkflow,
) -> None:
    output_data = {
        "entity_group": "MISC",
        "score": 0.9,
    }
    postprocessed_data = token_classification_workflow.do_postprocessing(
        output_data, output_data
    )
    assert postprocessed_data == output_data


def test_do_setup_summarization(
    summarization_workflow: HFInferenceClientWorkflow,
) -> None:
    summarization_workflow.setup()
    assert summarization_workflow.client is not None
    assert summarization_workflow.task_argspec is not None


def test_do_preprocessing_summarization(
    summarization_workflow: HFInferenceClientWorkflow,
) -> None:
    input_data = {"text": "Hello, world!"}
    preprocessed_data = summarization_workflow.do_preprocessing(input_data)
    assert preprocessed_data.model_dump()["text"] == input_data["text"]


def test_do_run_model_summarization(
    summarization_workflow: HFInferenceClientWorkflow,
) -> None:
    summarization_workflow.setup()
    min_length_tokens = 28
    max_length_tokens = 56
    summarization_config = HFSummarizationConfig(
        min_length=min_length_tokens,
        max_length=max_length_tokens,
    )
    input_text = """Artificial Intelligence has the capacity to positively impact
                humanity but the infrastructure in which it is being
                built is flawed. Permissioned and centralized APIs, lack of privacy
                and computational integrity, lack of censorship resistance — all
                risking the potential AI can unleash. Ritual is the network for
                open AI infrastructure. We build groundbreaking, new architecture
                on a crowdsourced governance layer aimed to handle safety, funding,
                alignment, and model evolution.
            """
    input_data = HFSummarizationInferenceInput(
        text=input_text,
        parameters=summarization_config,  # type: ignore
    )
    output_data = summarization_workflow.do_run_model(input_data)
    assert len(output_data.get("output")) > min_length_tokens  # type: ignore
    assert len(output_data.get("output")) < len(input_text)  # type: ignore


def test_do_postprocessing_summarization(
    summarization_workflow: HFInferenceClientWorkflow,
) -> None:
    output_data = {"summary": "Ritual's AI x Crypto stack is awesome!"}
    postprocessed_data = summarization_workflow.do_postprocessing(
        output_data, output_data
    )
    assert postprocessed_data == output_data


def test_do_setup_text_generation(
    text_generation_workflow: HFInferenceClientWorkflow,
) -> None:
    text_generation_workflow.setup()
    assert text_generation_workflow.client is not None
    assert text_generation_workflow.task_argspec is not None


def test_do_preprocessing_text_generation(
    text_generation_workflow: HFInferenceClientWorkflow,
) -> None:
    input_data = {"prompt": "How to run run ML inference on-chain?"}
    preprocessed_data = text_generation_workflow.do_preprocessing(input_data)
    assert preprocessed_data.model_dump()["prompt"] == input_data["prompt"]


def test_do_run_model_text_generation(
    text_generation_workflow: HFInferenceClientWorkflow,
) -> None:
    text_generation_workflow.setup()
    input_data = HFTextGenerationInferenceInput(
        prompt="Ritual's AI x Crypto stack is awesome!",
    )
    output_data = text_generation_workflow.do_run_model(input_data)
    assert len(output_data.get("output")) > 0  # type: ignore


def test_do_postprocessing_text_generation(
    text_generation_workflow: HFInferenceClientWorkflow,
) -> None:
    output_data = {"text": "Ritual's AI x Crypto stack is awesome!"}
    postprocessed_data = text_generation_workflow.do_postprocessing(
        output_data, output_data
    )
    assert postprocessed_data == output_data

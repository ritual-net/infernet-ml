"""
Module containing a TGI Inference Workflow object.
"""

import json
import os
from typing import Any, Optional, Union, cast

from pydantic import BaseModel
from retry import retry
from text_generation import Client  # type: ignore
from text_generation.errors import BadRequestError  # type: ignore
from text_generation.errors import (
    GenerationError,
    IncompleteGenerationError,
    NotFoundError,
    NotSupportedError,
    OverloadedError,
    RateLimitExceededError,
    ShardNotReadyError,
    ShardTimeoutError,
    UnknownError,
    ValidationError,
)

from infernet_ml.workflows.exceptions import ServiceException
from infernet_ml.workflows.inference.base_inference_workflow import (
    BaseInferenceWorkflow,
)

TGI_REQUEST_TRIES: int = json.loads(os.getenv("TGI_REQUEST_TRIES", "3"))
TGI_REQUEST_DELAY: Union[int, float] = json.loads(os.getenv("TGI_REQUEST_DELAY", "3"))
TGI_REQUEST_MAX_DELAY: Optional[Union[int, float]] = json.loads(
    os.getenv("TGI_REQUEST_MAX_DELAY", "null")
)
TGI_REQUEST_BACKOFF: Union[int, float] = json.loads(
    os.getenv("TGI_REQUEST_BACKOFF", "2")
)
TGI_REQUEST_JITTER: Union[tuple[float, float], float] = (
    jitter
    if isinstance(
        jitter := json.loads(os.getenv("TGI_REQUEST_JITTER", "[0.5,1.5]")), float
    )
    else tuple(jitter)
)


class TgiInferenceRequest(BaseModel):
    """
    Represents an TGI Inference Request
    """

    text: str  # query to the LLM backend


class TGIClientInferenceWorkflow(BaseInferenceWorkflow):
    """
    Workflow object for requesting LLM inference on TGI-compliant inference servers.
    """

    def __init__(
        self, server_url: str, timeout: int = 30, **inference_params: dict[str, Any]
    ) -> None:
        """
        constructor. Any named arguments passed to LLM during inference.

        Args:
            server_url (str): url of inference server
        """
        super().__init__()
        self.client: Client = Client(server_url, timeout=timeout)
        self.inference_params: dict[str, Any] = inference_params
        # dummy call to fail fast if client is misconfigured
        self.client.generate("hello", **inference_params)

    def do_setup(self) -> bool:
        """
        no specific setup needed
        """
        return True

    def do_preprocessing(self, input_data: dict[str, Any]) -> str:
        """
        Implement any preprocessing of the raw input.
        For example, you may want to append additional context.
        By default, returns the value associated with the text key in a dictionary.

        Args:
            input_data (Union[dict[str]]): raw input
        Returns:
            str: transformed user input prompt
        """
        TgiInferenceRequest.model_validate(input_data)
        return str(input_data["text"])

    def do_postprocessing(
        self, input_data: dict[str, Any], gen_text: str
    ) -> Union[str, dict[str, Any]]:
        """
        Implement any postprocessing here. For example, you may need to return
        additional data.

        Args:
            input_data (Union[dict[str]]): raw input
            gen_text (str): str result from LLM model
        Returns:
            Any: transformation to the gen_text
        """

        return gen_text

    @retry(
        exceptions=(
            ShardNotReadyError,
            ShardTimeoutError,
            RateLimitExceededError,
            OverloadedError,
        ),
        tries=TGI_REQUEST_TRIES,
        delay=TGI_REQUEST_DELAY,
        max_delay=TGI_REQUEST_MAX_DELAY,
        backoff=TGI_REQUEST_BACKOFF,
        jitter=TGI_REQUEST_JITTER,
    )
    def generate_inference(self, preprocessed_data: str) -> str:
        """use tgi client to generate inference.
        Args:
            preprocessed_data (str): input to tgi

        Returns:
            str: output of tgi inference
        """
        return cast(
            str,
            self.client.generate(
                preprocessed_data, **self.inference_params
            ).generated_text,
        )

    def do_run_model(self, preprocessed_data: str) -> str:
        """
        Inference implementation. Generally,
        you should not need to change this implementation
        directly, as the code already implements calling
        an LLM server.

        Instead, you can perform any preprocessing or
        post processing in the relevant abstract methods.

        Args:
            dict (str): user input

        Returns:
            Any: result of inference
        """
        try:
            return self.generate_inference(preprocessed_data)
        except (
            BadRequestError,
            GenerationError,
            IncompleteGenerationError,
            NotFoundError,
            NotSupportedError,
            OverloadedError,
            RateLimitExceededError,
            ShardNotReadyError,
            ShardTimeoutError,
            UnknownError,
            ValidationError,
        ) as e:
            # we catch expected service exceptions and return ServiceException
            # this is so we can handle unexpected vs. expected exceptions
            # downstream
            raise ServiceException(e) from e

    def do_generate_proof(self) -> Any:
        """
        raise error by default
        """
        raise NotImplementedError

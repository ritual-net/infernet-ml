"""
Module containing a CSS (Closed Source Software) Inference Workflow object.

See css_mux.py for more details on supported closed source libraries.
In addition to the constructor arguments "provider" and "endpoint", note the
appropriate API key needs to be specified in environment variables.

"""

import json
import logging
import os
from typing import Any, Optional, Union

from retry import retry

from infernet_ml.utils.css_mux import css_mux, validate
from infernet_ml.utils.service_models import CSSRequest
from infernet_ml.workflows.exceptions import RetryableException
from infernet_ml.workflows.inference.base_inference_workflow import (
    BaseInferenceWorkflow,
)

CSS_REQUEST_TRIES: int = json.loads(os.getenv("CSS_REQUEST_TRIES", "3"))
CSS_REQUEST_DELAY: Union[int, float] = json.loads(os.getenv("CSS_REQUEST_DELAY", "3"))
CSS_REQUEST_MAX_DELAY: Optional[Union[int, float]] = json.loads(
    os.getenv("CSS_REQUEST_MAX_DELAY", "null")
)
CSS_REQUEST_BACKOFF: Union[int, float] = json.loads(
    os.getenv("CSS_REQUEST_BACKOFF", "2")
)
CSS_REQUEST_JITTER: Union[tuple[float, float], float] = (
    jitter
    if isinstance(
        jitter := json.loads(os.getenv("CSS_REQUEST_JITTER", "[0.5,1.5]")), float
    )
    else tuple(jitter)
)


class CSSInferenceWorkflow(BaseInferenceWorkflow):
    """
    Base workflow object for closed source LLM inference models.
    """

    def __init__(self, provider: str, endpoint: str, **inference_params: Any) -> None:
        """
        constructor. Any named arguments passed to closed source LLM during inference.

        Args:
            server_url (str): url of inference server
        """
        super().__init__()
        self.inference_params: dict[str, Any] = inference_params
        # default inference params with provider endpoint and model
        inference_params["provider"] = provider
        inference_params["endpoint"] = endpoint
        # validate provider and endpoint
        validate(provider, endpoint)

    def do_setup(self) -> bool:
        """
        no specific setup needed
        """
        return True

    def do_preprocessing(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Implement any preprocessing of the raw user input.
        For example, you may want to append additional context or parameters to the
        closed source model prior to querying.
        By default, returns the input dictionary associated with the user input.

        Args:
            input_data (Union[dict[str]]): raw user input

        Returns:
            dict[str, Any]: transformed user input prompt
        """
        preprocess_dict: dict[str, Any] = self.inference_params.copy()
        preprocess_dict.update(input_data)
        return preprocess_dict

    def do_postprocessing(
        self, input_data: dict[str, Any], gen_text: str
    ) -> Union[str, dict[str, Any]]:
        """
        Implement any postprocessing here. For example, you may need to return
        additional data.

        Args:
            input_data (dict[str, Any]): original input data from client
            gen_text (str): str result from closed source LLM model

        Returns:
            Any: transformation of the gen_text
        """

        return gen_text

    @retry(
        tries=CSS_REQUEST_TRIES,
        delay=CSS_REQUEST_DELAY,
        max_delay=CSS_REQUEST_MAX_DELAY,
        backoff=CSS_REQUEST_BACKOFF,
        jitter=CSS_REQUEST_JITTER,
        exceptions=(RetryableException,),
    )
    def do_run_model(
        self, preprocessed_data: dict[str, Any]
    ) -> Union[str, dict[str, Any]]:
        """
        Inference implementation. Generally, you should not need to change this
        implementation directly, as the code already implements calling a closed source
         LLM server.

        Instead, you can perform any preprocessing or post processing in the relevant
        abstract methods.

        Args:
            input_data dict (str): user input

        Returns:
            Union[str, dict[str, Any]]: result of inference
        """

        preprocessed_data_parsed: CSSRequest = CSSRequest(**preprocessed_data)
        logging.info(
            f"querying {preprocessed_data['provider']} with {type(preprocessed_data_parsed)} [{preprocessed_data_parsed}]..."  # noqa:E501
        )
        # TODO: consider async
        return css_mux(preprocessed_data["provider"], preprocessed_data_parsed)

    def do_generate_proof(self) -> Any:
        """
        raise error by default
        """
        raise NotImplementedError

"""
HF Inference Service entry point
"""

import json
import logging
from typing import Any, Optional, Union, cast

import eth_abi.exceptions
from eth_abi import decode  # type: ignore
from eth_abi import encode  # type: ignore
from pydantic import ValidationError as PydValError
from quart import Quart, abort
from quart import request as req
from quart.utils import run_sync
from werkzeug.exceptions import HTTPException

from infernet_ml.utils.service_models import InfernetInput, InfernetInputSource
from infernet_ml.workflows.exceptions import ServiceException
from infernet_ml.workflows.inference.hf_inference_client_workflow import (
    HFInferenceClientWorkflow,
)

SERVICE_CONFIG_PREFIX = "HF_INF"


def create_app(
    task: str, model: Optional[str] = None, test_config: Optional[dict[str, Any]] = None
) -> Quart:
    """Huggingface Inference Service application factory

    Args:
        task (str): Task to be performed by the service. Supported tasks are:
         text_classification, text_generation, summarization, conversational,
          token_classification
        model (Optional[str]): Model to be used for inference or None to auto
          deduce. Defaults to None.
        test_config (Optional[dict[str, Any]], optional): Configs for testing.
          overrides env vars. Defaults to None.

    Returns:
        Quart: Quart App instance

    Raises:
        ImportError: thrown if error loading the workflow
        PydValError: thrown if error during input validation
    """
    app: Quart = Quart(__name__)
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_prefixed_env(prefix=SERVICE_CONFIG_PREFIX)
    else:
        # load the test config if passed in
        app.config.update(test_config)
    # Override task and model if config is set
    task = app.config.get("TASK", "") or task
    model = app.config.get("MODEL", "") or model
    WORKFLOW = HFInferenceClientWorkflow(
        task,
        model,
        token=app.config.get("HF_TOKEN"),
    )
    # Setup workflow
    logging.info(f"Setting up Huggingface Inference Workflow for {task} task")
    WORKFLOW.setup()

    @app.route("/")
    async def index() -> str:
        return f"Infernet-ML HuggingFace Model Inference Service for {task} task"

    @app.route("/service_output", methods=["POST"])
    @app.route("/inference", methods=["POST"])
    async def inference() -> Union[str, dict[str, Any]]:
        """Invokes inference backend HF client. Expects json/application data,
        formatted according to the InferenceRequest schema.
        Returns:
            dict: Inference result
        """
        result: Union[str, dict[str, Any]]
        if req.method == "POST" and (data := await req.get_json()):
            try:
                ## Validate input data using pydantic model
                inf_input = InfernetInput(**data)
                match inf_input:
                    case InfernetInput(
                        source=InfernetInputSource.OFFCHAIN, data=input_data
                    ):
                        logging.info(f"Received Offchain Request: {input_data}")
                        result = await run_sync(WORKFLOW.inference)(
                            input_data=input_data
                        )

                    case InfernetInput(
                        source=InfernetInputSource.CHAIN, data=input_data
                    ):
                        logging.info(f"Received Onchain Request:{input_data}")
                        # Decode input data from eth_abi bytes32 to string
                        abi_types = ["bytes[]"]
                        input_data_decoded = decode(
                            abi_types, bytes.fromhex(cast(str, input_data))
                        )

                        # Decode input_data from bytes to JSON object
                        input_data_list = []  # mini-batch
                        for item in input_data_decoded:
                            item = item[0] if len(item) == 1 else item
                            try:
                                input_data_list.append(json.loads(item.decode("utf-8")))
                            except json.JSONDecodeError:
                                logging.error(f"Failed to decode input data: {item}")
                                abort(400, "Failed to decode input data")

                        logging.info(f"Decoded input data: {input_data_list}")
                        # Send parsed input_data for inference
                        result_native = await run_sync(WORKFLOW.inference)(
                            input_data=input_data_list[0]  # batch_size=1
                        )

                        try:
                            # Encode result to eth_abi bytes[] for onchain response
                            result_json = json.dumps(result_native).encode("utf-8")

                            abi_types = ["bytes[]"]
                            result_encoded = encode(abi_types, [[result_json]])
                            result = result_encoded.hex()
                            logging.info(f"Encoded result: {result}")
                        except eth_abi.exceptions.EncodingTypeError:
                            logging.error(f"Failed to encode {result} using eth_abi")
                            abort(500, "Failed to encode result using eth_abi")
                    case _:
                        raise PydValError(
                            "Invalid InferentInput type: expected mapping for offchain input type"  # noqa: E501
                        )

                logging.info(f"Received result from workflow: {result}")
                return result
            except ServiceException as e:
                abort(500, e)
            except PydValError as e:
                abort(400, e)

        abort(400, "Invalid method or data: Only POST supported with json data")

    @app.errorhandler(HTTPException)
    def handle_exception(e: Any) -> Any:
        """Return JSON instead of HTML for HTTP errors."""
        # start with the correct headers and status code from the error
        response = e.get_response()

        # replace the body with JSON
        response.data = json.dumps(
            {
                "code": str(e.code),
                "name": str(e.name),
                "description": str(e.description),
            }
        )

        response.content_type = "application/json"
        return response

    return app


if __name__ == "__main__":
    app = create_app(task="text_classification")
    app.run()

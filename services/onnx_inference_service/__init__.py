"""
this module serves as the driver for the tgi inference service.
"""

import json
import logging
import os
from typing import Any, cast

import numpy as np
from eth_abi.abi import decode
from pydantic import ValidationError as PydValError
from quart import Quart, abort
from quart import request as req
from quart.json.provider import DefaultJSONProvider
from quart.utils import run_sync
from werkzeug.exceptions import HTTPException

from infernet_ml.utils.model_loader import ModelSource
from infernet_ml.utils.service_models import InfernetInput, InfernetInputSource
from infernet_ml.workflows.exceptions import ServiceException
from infernet_ml.workflows.inference.onnx_inference_workflow import (
    ONNXInferenceWorkflow,
)


class NumpyJsonEncodingProvider(DefaultJSONProvider):
    @staticmethod
    def default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            # Convert NumPy arrays to list
            return obj.tolist()
        # fallback to default JSON encoding
        return DefaultJSONProvider.default(obj)


FLOAT_DECIMALS = 9


def create_app(test_config: dict[str, Any]) -> Quart:
    """
    application factory for the ONNX Inference Service

    Raises:
        PydValError: thrown if error during input validation

    Returns:
        Quart: Quart App instance
    """
    Quart.json_provider_class = NumpyJsonEncodingProvider
    app: Quart = Quart(__name__)
    app.config.from_mapping()

    logging.info(
        "setting up ONNX inference",
    )

    MODEL_ARGS = os.getenv("MODEL_ARGS", "{}")
    if MODEL_ARGS[0] == "'" or MODEL_ARGS[0] == '"':
        MODEL_ARGS = MODEL_ARGS[1:-1]

    app_config = test_config or {
        "kwargs": {
            "output_names": os.getenv("ONNX_OUTPUT_NAMES", "output").split(","),
            "model_source": ModelSource(os.getenv("MODEL_SOURCE", "LOCAL")),
            "model_args": json.loads(MODEL_ARGS),
        }
    }
    kwargs = app_config["kwargs"]

    WORKFLOW = ONNXInferenceWorkflow(**kwargs).setup()

    @app.route("/")
    async def index() -> str:
        return "ONNX Inference Service!"

    @app.route("/service_output", methods=["POST"])
    async def inference() -> dict[str, Any]:
        """
        implements inference. Expects json/application data,
        formatted according to the InferenceRequest schema.
        Returns:
            dict: inference result
        """
        if req.method == "POST" and (data := await req.get_json()):
            # we will get the file from the request
            try:
                # load data into model for validation
                inf_input = InfernetInput(**data)
                match inf_input:
                    case InfernetInput(source=InfernetInputSource.OFFCHAIN, data=data):
                        logging.info("received Offchain Request: %s", data)
                        # send parsed output back
                        input_data = data
                    case InfernetInput(source=InfernetInputSource.CHAIN, data=data):
                        logging.info("received On-chain Request: %s", data)
                        # decode web3 abi.encode(uint64, uint64, uint64, uint64)
                        (a, b, c, d) = decode(
                            ["uint64", "uint64", "uint64", "uint64"],
                            bytes.fromhex(cast(str, data)),
                        )
                        input_data = {
                            "input": [[a / 10**FLOAT_DECIMALS for a in [a, b, c, d]]]
                        }
                    case _:
                        raise PydValError(
                            "Invalid InferentInput type: expected mapping for offchain input type"  # noqa: E501
                        )
                print("input_data", input_data)
                result: dict[str, Any] = await run_sync(WORKFLOW.inference)(input_data)

                logging.info("received result from workflow: %s", result)
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

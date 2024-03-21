"""
This module serves as the driver for torch infernet_ml inference service.
"""

import importlib
import json
import logging
from typing import Any, Optional, Type, Union, cast

from eth_abi.abi import decode
from quart import Quart, abort
from quart import request as req
from werkzeug.exceptions import HTTPException

from infernet_ml.utils.decode import decode_vector
from infernet_ml.utils.service_models import InfernetInput, InfernetInputSource
from infernet_ml.workflows.inference.base_inference_workflow import (
    BaseInferenceWorkflow,
)

SERVICE_PREFIX = "TORCH_INF"


def get_workflow_class(
    full_class_path: str,
) -> Optional[Type[BaseInferenceWorkflow]]:
    """Return a BaseInferenceWorkflow instance from a string reference

    Args:
        full_class_path (str): class path

    Returns:
        Optional[Type[BaseInferenceWorkflow]]: returns None if
        error loading
    """
    module_name, class_name = full_class_path.rsplit(".", 1)
    try:
        class_ = None
        module_ = importlib.import_module(module_name)
        try:
            class_ = getattr(module_, class_name)
            if not issubclass(class_, BaseInferenceWorkflow):
                logging.error(
                    "%s is not a subclass of BaseClassicInferenceWorkflow", class_
                )
                class_ = None
        except AttributeError:
            logging.error("Class does not exist")
    except ImportError:
        logging.error("Module does not exist")

    return class_


def create_app(test_config: Optional[dict[str, Any]] = None) -> Quart:
    """
    Factory function that creates and configures an instance
    of the Quart application

    Args:
        test_config (dict, optional): test config. Defaults to None.

    Returns:
        Quart: Quart App
    """
    app = Quart(__name__)
    app.config.from_mapping(
        # should be overridden by instance config
        WORKFLOW_CLASS="infernet_ml.workflows.inference.torch_inference_workflow.TorchInferenceWorkflow",
        WORKFLOW_POSITIONAL_ARGS=[],
        WORKFLOW_KW_ARGS={},
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_prefixed_env(prefix=SERVICE_PREFIX)
    else:
        # load the test config if passed in
        app.config.update(test_config)

    CLASSIC_WORKFLOW_CLASS = app.config["WORKFLOW_CLASS"]
    CLASSIC_WORKFLOW_POSITIONAL_ARGS = app.config["WORKFLOW_POSITIONAL_ARGS"]
    CLASSIC_WORKFLOW_KW_ARGS = app.config["WORKFLOW_KW_ARGS"]
    logging.info(
        "Loading %s %s %s: %s",
        CLASSIC_WORKFLOW_CLASS,
        CLASSIC_WORKFLOW_POSITIONAL_ARGS,
        CLASSIC_WORKFLOW_KW_ARGS,
        app.config,
    )

    clazz = get_workflow_class(CLASSIC_WORKFLOW_CLASS)

    if clazz is None:
        raise ImportError(
            f"Unable to import specified Workflow class {CLASSIC_WORKFLOW_CLASS}"
        )

    # create workflow instance from class, using specified arguments
    WORKFLOW: BaseInferenceWorkflow = clazz(
        *CLASSIC_WORKFLOW_POSITIONAL_ARGS, **CLASSIC_WORKFLOW_KW_ARGS
    )

    # setup workflow
    WORKFLOW.setup()

    @app.route("/")
    def index() -> str:
        return f"Torch ML Inference Service : {WORKFLOW.__class__}"

    @app.route("/service_output", methods=["POST"])
    async def inference() -> Union[str, dict[str, Any]]:
        """
        Performs inference from the model.
        """

        if req.method == "POST":
            # we will get the json from the request.
            # Type information read from model schema

            # get data as json
            data: Optional[dict[str, Any]] = await req.get_json()
            if data:
                input: InfernetInput = InfernetInput(**data)

                inference_input = input.data
                if input.source == InfernetInputSource.CHAIN:
                    decoded = decode(
                        ["int256[]"], bytes.fromhex(cast(str, input.data))
                    )[0]
                    # we assume double dtype if onchain

                    # TODO: consider making this more generic
                    inference_input = {
                        "values": [decode_vector(decoded)],
                        "dtype": "double",
                    }
                result = WORKFLOW.inference(inference_input)
                return {
                    "data": [o.detach().numpy().reshape([-1]).tolist() for o in result]
                }
            else:
                abort(400, "MIME type application/json expected")
        else:
            abort(400, "only POST method supported for this endpoint")

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

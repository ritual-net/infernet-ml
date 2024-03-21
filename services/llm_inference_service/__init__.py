"""
this module serves as the driver for the llm inference service.
"""

import importlib
import json
import logging
from typing import Any, Optional, Type, Union

from pydantic import ValidationError as PydValError
from quart import Quart, abort
from quart import request as req
from quart.utils import run_sync
from werkzeug.exceptions import HTTPException

from infernet_ml.utils.service_models import InfernetInput, InfernetInputSource
from infernet_ml.workflows.exceptions import ServiceException
from infernet_ml.workflows.inference.css_inference_workflow import CSSInferenceWorkflow
from infernet_ml.workflows.inference.tgi_client_inference_workflow import (
    TGIClientInferenceWorkflow,
)

SERVICE_PREFIX = "LLM_INF"


def get_workflow_class(
    full_class_path: str,
) -> Optional[Union[Type[TGIClientInferenceWorkflow], Type[CSSInferenceWorkflow]]]:
    """
    Returns a TGIClientInferenceWorkflow or CSSInferenceWorkflow instance from a class
    path string.

    Args:
        full_class_path (str): class to load

    Returns:
        Optional[
            Union[Type[TGIClientInferenceWorkflow],Type[CSSInferenceWorkflow]]
        ]: None
        if error loading the class
    """
    module_name, class_name = full_class_path.rsplit(".", 1)
    try:
        class_ = None
        module_ = importlib.import_module(module_name)
        try:
            class_ = getattr(module_, class_name)
            if not (
                issubclass(class_, CSSInferenceWorkflow)
                or issubclass(class_, TGIClientInferenceWorkflow)
            ):
                logging.error(
                    "%s is not a subclass of CSSInferenceWorkflow or "
                    + "TGIClientInferenceWorkflow",
                    class_,
                )
                class_ = None
        except AttributeError:
            logging.error("Class does not exist")
    except ImportError as e:
        logging.error("Module does not exist: %s", e)

    return class_


def create_app(test_config: Optional[dict[str, Any]] = None) -> Quart:
    """application factory for the LLM Inference Service

    Args:
        test_config (Optional[dict[str, Any]], optional): Defaults to None.

    Raises:
        ImportError: thrown if error loading the workflow
        PydValError: thrown if error duing input validation

    Returns:
        Quart: Quart App instance
    """
    app: Quart = Quart(__name__)
    app.config.from_mapping(
        # should be overridden by instance config
        WORKFLOW_CLASS="infernet_ml.workflows.inference.tgi_client_inference_workflow.TGIClientInferenceWorkflow",
        WORKFLOW_POSITIONAL_ARGS=["http://server_url_here"],
        WORKFLOW_KW_ARGS={},
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_prefixed_env(prefix=SERVICE_PREFIX)
    else:
        # load the test config if passed in
        app.config.update(test_config)

    LLM_WORKFLOW_CLASS = app.config["WORKFLOW_CLASS"]
    LLM_WORKFLOW_POSITIONAL_ARGS = app.config["WORKFLOW_POSITIONAL_ARGS"]
    LLM_WORKFLOW_KW_ARGS = app.config["WORKFLOW_KW_ARGS"]

    logging.info(
        "%s %s %s",
        LLM_WORKFLOW_CLASS,
        LLM_WORKFLOW_POSITIONAL_ARGS,
        LLM_WORKFLOW_KW_ARGS,
    )

    clazz = get_workflow_class(LLM_WORKFLOW_CLASS)

    if clazz is None:
        raise ImportError(
            f"Unable to import specified Workflow class {LLM_WORKFLOW_CLASS}"
        )

    # create workflow instance from class, using specified arguments
    LLM_WORKFLOW = clazz(*LLM_WORKFLOW_POSITIONAL_ARGS, **LLM_WORKFLOW_KW_ARGS)

    # setup workflow
    LLM_WORKFLOW.setup()

    @app.route("/")
    async def index() -> str:
        """Default index page
        Returns:
            str: simple heading
        """
        return f"<p>Lightweight LLM Inference Service to {clazz.__name__}</p>"

    @app.route("/service_output", methods=["POST"])
    @app.route("/inference", methods=["POST"])
    async def inference() -> Union[str, dict[str, Any]]:
        """implements inference. Expects json/application data,
        formatted according to the InferenceRequest schema.
        Returns:
            dict: inference result
        """
        if req.method == "POST" and (data := await req.get_json()):
            # we will get the file from the request
            try:
                ## load data into model for validation
                inf_input = InfernetInput(**data)
                match inf_input:
                    case InfernetInput(
                        source=InfernetInputSource.OFFCHAIN, data=input_data
                    ):
                        logging.info("received Offchain Request: %s", input_data)

                        ## send parsed output back
                        result: Union[str, dict[str, Any]] = await run_sync(
                            LLM_WORKFLOW.inference
                        )(input_data=input_data)

                        logging.info("recieved result from workflow: %s", result)
                        return result
                    case _:
                        raise PydValError(
                            "Invalid InferentInput type: expected mapping for offchain input type"  # noqa: E501
                        )
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

"""
workflow class for onnx inference workflows.
"""
import logging
from typing import Any, Optional

import onnx
import torch
from onnxruntime import InferenceSession  # type: ignore

from infernet_ml.utils.model_loader import ModelSource, load_model
from infernet_ml.workflows.inference.base_inference_workflow import (
    BaseInferenceWorkflow,
)

logger: logging.Logger = logging.getLogger(__name__)


class ONNXInferenceWorkflow(BaseInferenceWorkflow):
    """
    Inference workflow for ONNX models.
    """

    ort_session: InferenceSession

    def __init__(
        self,
        model_source: ModelSource = ModelSource.LOCAL,
        model_args: Optional[dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.output_names = kwargs.get("output_names", [])
        self.model_source = model_source
        self.model_args = model_args or {}

    def do_setup(self) -> "ONNXInferenceWorkflow":
        """set up here (if applicable)."""
        return self.load_model()

    def do_preprocessing(self, input_data: dict[Any, Any]) -> Any:
        return {k: torch.Tensor(input_data[k]).numpy() for k in input_data}

    def load_model(self) -> "ONNXInferenceWorkflow":
        """
        Loads and checks the ONNX model. if called will attempt to download latest
        version of model. If the check is successful it will start an inference session.

        Returns:
            bool: True on completion of loading model
        """
        model_path = load_model(self.model_source, **self.model_args)

        # check model
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)

        # start the inference session
        self.ort_session = InferenceSession(model_path)
        return self

    def do_run_model(self, input_feed: dict[str, Any]) -> Any:
        outputs = self.ort_session.run(self.output_names, input_feed)
        return outputs

    def do_postprocessing(self, input_data: Any, output_data: Any) -> Any:
        """
        Simply return the output from the model. Post-processing can be implemented
        by overriding this method.
        """
        return output_data

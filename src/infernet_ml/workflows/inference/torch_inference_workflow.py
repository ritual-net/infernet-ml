"""
workflow  class for torch inference workflows.
"""

import logging
import os
from typing import Any, Optional, cast

# This noinspection is required otherwise IDE import optimizations would remove the
# import statement, which would prevent scikit-learn models to be included in the scope.
# noinspection PyUnresolvedReferences
import torch
import torch.jit

from infernet_ml.utils.model_loader import ModelSource, load_model
from infernet_ml.workflows.inference.base_inference_workflow import (
    BaseInferenceWorkflow,
)

logger: logging.Logger = logging.getLogger(__name__)

# whether or not to use torch script
USE_JIT = os.getenv("USE_JIT", "False").lower() in ("true", "1", "t")


# dtypes we support for conversion to corresponding torch types.
DTYPES = {
    "float": torch.float,
    "double": torch.double,
    "cfloat": torch.cfloat,
    "cdouble": torch.cdouble,
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "short": torch.short,
    "int": torch.int,
    "long": torch.long,
    "bool": torch.bool,
}


class TorchInferenceWorkflow(BaseInferenceWorkflow):
    """
    Inference workflow for Torch based models. models are loaded using the default
    torch pickling by default(i.e. torch.load). This can be changed to use torch script
     (torch.jit) if the USE_JIT environment variable is enabled.

    By default, uses hugging face to download the model file, which requires
    HUGGING_FACE_HUB_TOKEN to be set in the env vars to access private models. if
    the USE_ARWEAVE env var is set to true, will attempt to download models via
    Arweave, reading env var ALLOWED_ARWEAVE_OWNERS as well.
    """

    def __init__(
        self,
        model_source: ModelSource = ModelSource.LOCAL,
        model_args: Optional[dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model: Optional[torch.nn.Module] = None
        self.model_source = model_source
        self.model_args = model_args or {}

    def do_setup(self) -> Any:
        """set up here (if applicable)."""
        return self.load_model()

    def load_model(self) -> bool:
        """loads the model. if called will attempt to download latest version of model
        based on the specified model source.

        Returns:
            bool: True on completion of loading model
        """

        model_path = load_model(self.model_source, **self.model_args)

        self.model = torch.jit.load(model_path) if USE_JIT else torch.load(model_path)  # type: ignore  # noqa: E501

        # turn on inference mode
        self.model.eval()  # type: ignore

        logging.info("model loaded")

        return True

    def do_preprocessing(self, input_data: dict[str, Any]) -> torch.Tensor:
        # lookup dtype from str
        dtype = DTYPES.get(input_data["dtype"], None)
        values = input_data["values"]
        return torch.tensor(values, dtype=dtype)

    def do_run_model(self, preprocessed_data: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not loaded, call setup() first")
        model_result = cast(torch.Tensor, self.model(preprocessed_data))
        return model_result

    def do_postprocessing(self, input_data: Any, output_data: Any) -> Any:
        return output_data

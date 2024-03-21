"""
Module for the generic HuggingFace Inference Workflow object.
The goal of this module is to provide a generic interface to run inference on any
 Hugging Face models for any of the supported tasks across the domains.
"""

import inspect
import logging
from typing import Any, Optional

from huggingface_hub import InferenceClient  # type: ignore[import-untyped]
from pydantic import ValidationError

from infernet_ml.utils.hf_types import (
    HFClassificationInferenceInput,
    HFInferenceInput,
    HFSummarizationInferenceInput,
    HFTextGenerationInferenceInput,
)
from infernet_ml.workflows.inference.base_inference_workflow import (
    BaseInferenceWorkflow,
)

# Dict of task_id to task_name grouped by domain based on https://huggingface.co/tasks
AVAILABLE_DOMAIN_TASKS = {
    "Audio": {
        "audio_classification": "Audio Classification",
        "automatic_speech_recognition": "Automatic Speech Recognition",
        "text_to_speech": "Text to Speech",
    },
    "Computer Vision": {
        "image_classification": "Image Classification",
        "image_segmentation": "Image Segmentation",
        "image_to_image": "Image to Image",
        "image_to_text": "Image to Text",
        "object_detection": "Object Detection",
        "text_to_image": "Text to Image",
        "zero_shot_image_classification": "Zero-Shot Image Classification",
    },
    "Multimodal": {
        "document_question_answering": "Document Question Answering",
        "visual_question_answering": "Visual Question Answering",
    },
    "NLP": {
        "conversational": "Conversational",
        "feature_extraction": "Feature Extraction",
        "fill_mask": "Fill Mask",
        "question_answering": "Question Answering",
        "sentence_similarity": "Sentence Similarity",
        "summarization": "Summarization",
        "table_question_answering": "Table Question Answering",
        "text_classification": "Text Classification",
        "text_generation": "Text Generation",
        "token_classification": "Token Classification",
        "translation": "Translation",
        "zero_shot_classification": "Zero-Shot Classification",
        "tabular_classification": "Tabular Classification",
        "tabular_regression": "Tabular Regression",
    },
}
# Maintain a list of supported tasks
SUPPORTED_TASKS = [
    "text_generation",
    "text_classification",
    "token_classification",
    "summarization",
]

# Logger for the module
logger = logging.getLogger(__name__)


class HFInferenceClientWorkflow(BaseInferenceWorkflow):
    """
    Inference workflow for models available through Huggingface Hub.
    """

    def __init__(
        self, task: str, model: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> None:
        """
        Initialize the Huggingface Inference Workflow object

        Args:
            task (str): HF supported task id
            model (str): model id on Huggingface Hub to be used for inference
        Raises:
            ValueError: if task is not supported
        """
        super().__init__(*args, **kwargs)
        # Ensure task is one of supported tasks
        logger.debug(f"Initializing Huggingface Inference Workflow for task {task}")
        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Task {task} is not supported. Supported tasks are {SUPPORTED_TASKS}"
            )
        logger.debug(f"Task {task} is supported")
        self.task_id = task
        self.model_id = model
        self.inference_params: dict[str, Any] = {}

    def do_setup(self) -> bool:
        """
        Setup the inference client
        """
        self.client = InferenceClient(token=self.kwargs.get("token"))
        self.task = self.client.__getattribute__(self.task_id)
        self.task_argspec = inspect.getfullargspec(self.task)
        done = (
            isinstance(self.client, InferenceClient)
            and self.task is not None
            and self.task_argspec is not None
        )
        logger.debug(f"Setup done: {done}")
        return done

    def do_preprocessing(self, input_data: dict[str, Any]) -> HFInferenceInput:
        # Handle task specific input data
        try:
            preprocessed_data: HFInferenceInput
            match self.task_id:
                case "text_classification":
                    preprocessed_data = HFClassificationInferenceInput(**input_data)
                case "token_classification":
                    preprocessed_data = HFClassificationInferenceInput(**input_data)
                case "summarization":
                    preprocessed_data = HFSummarizationInferenceInput(**input_data)
                case "text_generation":
                    preprocessed_data = HFTextGenerationInferenceInput(**input_data)
                case _:
                    preprocessed_data = HFInferenceInput(**input_data)
        except ValidationError as e:
            raise ValueError(f"Invalid input data: {e} for {self.task_id} task")

        return preprocessed_data

    def do_run_model(self, preprocessed_data: HFInferenceInput) -> dict[str, Any]:
        """
        Perform inference on the input data

        Args:
            preprocessed_data (HFInferenceInput): preprocessed input data

        Returns:
            dict: output data from the inference call
        """
        output = self.task(**preprocessed_data.model_dump(), **self.inference_params)
        logger.debug(f"Output from inference call: {output}")
        return {"output": output}

    def do_postprocessing(
        self, input_data: Any, output: dict[str, Any]
    ) -> dict[str, Any]:
        # Postprocessing logic here
        return output

    def do_generate_proof(self) -> Any:
        raise NotImplementedError

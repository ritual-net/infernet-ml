"""
Module containing a TTS (text-to-speech) Inference Workflow object.
"""
import logging
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, model_validator

from infernet_ml.workflows.inference.base_inference_workflow import (
    BaseInferenceWorkflow,
)


class AudioInferenceResult(BaseModel):
    """
    Base workflow object for text-to-speech inference.
    """

    class Config:
        """since audio_array is a numpy array, we need to allow arbitrary types."""

        arbitrary_types_allowed = True

    """
    audio_array: numpy array containing the audio data. Most TTS models' output will be
    a numpy array.
    """
    audio_array: np.ndarray[Any, Any]

    @model_validator(mode="after")
    def check_is_array(self) -> "AudioInferenceResult":
        if not isinstance(self.audio_array, np.ndarray):
            raise ValueError("audio_array must be a numpy ndarray")
        return self


logger: logging.Logger = logging.getLogger(__name__)


class TTSInferenceWorkflow(BaseInferenceWorkflow):
    """
    Base workflow object for text-to-speech inference.
    """

    """
    Sample Rate: The sample rate of the audio data. This is required for writing the
    audio data to a wav file.
    """
    SAMPLE_RATE: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def do_postprocessing(
        self, input_data: Any, audio_array: torch.Tensor
    ) -> AudioInferenceResult:
        """
        Post-processing of the audio array. This method should be implemented by the
        child class. In most cases it will include generation of a wav file.
        """
        pass

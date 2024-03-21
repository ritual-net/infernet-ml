"""
Implementation of Suno's Bark TTS (text-to-speech) Inference Workflow.
"""
from typing import Any, Optional, Protocol, cast

import numpy
import torch
from pydantic import BaseModel
from transformers import AutoProcessor  # type: ignore
from transformers import BarkModel, BatchEncoding

from infernet_ml.workflows.inference.tts_inference_workflow import (
    AudioInferenceResult,
    TTSInferenceWorkflow,
)


class BarkProcessor(Protocol):
    """
    Type for the Suno Processor function. Used for type-safety.
    """

    def __call__(self, input_data: str, voice_preset: str) -> BatchEncoding:
        """
        Args:
            input_data (str): prompt to generate audio from
            voice_preset (str): voice to be used. There is a list of supported presets
            here: https://github.com/suno-ai/bark?tab=readme-ov-file#-voice-presets

        Returns:
            BatchEncoding: batch encoding of the input data
        """
        ...


class BarkWorkflowInput(BaseModel):
    # prompt to generate audio from
    prompt: str
    # voice to be used. There is a list of supported presets here:
    # here: https://github.com/suno-ai/bark?tab=readme-ov-file#-voice-presets
    voice_preset: Optional[str]


class BarkHFInferenceWorkflow(TTSInferenceWorkflow):
    """
    Implementation of Suno TTS Inference Workflow.
    """

    SAMPLE_RATE: int = 24_000
    model: BarkModel
    processor: BarkProcessor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # name of the model to be used. The allowed values are: suno/bark,
        # suno/bark-small (default: suno/bark)
        self.model_name = kwargs.get("model_name", "suno/bark")
        # default voice preset to be used. Refer to the link for the list of supported
        # presets (default: v2/en_speaker_6)
        # https://github.com/suno-ai/bark?tab=readme-ov-file#-voice-presets
        self.default_voice_preset = kwargs.get(
            "default_voice_preset", "v2/en_speaker_6"
        )
        # device to be used for inference. If cuda is available, it will be used,
        # else cpu will be used
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def do_setup(self) -> None:
        """
        Downloads the model from huggingface.
        Returns:
            bool: True on completion of loading model
        """
        self.model = BarkModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def do_preprocessing(self, input_data: BarkWorkflowInput) -> BatchEncoding:
        """
        Args:
            input_data (BarkWorkflowInput): input data to be preprocessed
        Returns:
            BatchEncoding: batch encoding of the input data
        """
        text = input_data.prompt
        voice_preset = input_data.voice_preset or self.default_voice_preset
        return self.processor(text, voice_preset=voice_preset).to(self.device)

    def inference(self, input_data: BarkWorkflowInput) -> AudioInferenceResult:  # type: ignore #noqa: E501
        """
        Override super class inference method to be annotated with the correct types.
        Args:
            input_data (str): prompt to generate audio from
        Returns:
            AudioInferenceResult: audio array
        """
        return cast(AudioInferenceResult, super().inference(input_data))

    def do_run_model(self, preprocessed_data: BatchEncoding) -> torch.Tensor:
        return cast(torch.Tensor, self.model.generate(**preprocessed_data))

    def do_postprocessing(
        self, input_data: Any, output: torch.Tensor
    ) -> AudioInferenceResult:
        """
        Converts the model output to a numpy array, which then can be used to save the
        audio file.
        Args:
            input_data(Any): original input data
            output (torch.Tensor): output tensor from the model

        Returns:
            AudioInferenceResult: audio array
        """
        audio_array: numpy.ndarray[Any, Any] = output.cpu().numpy().squeeze()
        return AudioInferenceResult(audio_array=audio_array)

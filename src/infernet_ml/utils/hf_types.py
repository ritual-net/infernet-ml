from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired


class HFInferenceInput(BaseModel):
    """Base class for input data"""

    model: Optional[str] = None


class HFClassificationInferenceInput(HFInferenceInput):
    """Input data for classification models"""

    text: str


class HFTextGenerationInferenceInput(HFInferenceInput):
    """Input data for text generation models

    Args:
        prompt (str): Prompt for text generation
        details (bool): Whether to return detailed output (tokens, probabilities,
            seed, finish reason, etc.)
        stream (bool): Whether to stream output. Only available for models
            running with the `text-generation-interface` backend.
        do_sample (bool): Whether to use logits sampling
        max_new_tokens (int): Maximum number of tokens to generate
        best_of (int): Number of best sequences to generate and return
            with highest token logprobs
        repetition_penalty (float): Repetition penalty for greedy decoding.
            1.0 is no penalty
        return_full_text (bool): Whether to preprend the prompt to
            the generated text
        seed (int): Random seed for generation sampling
        stop_sequences (str): Sequence to stop generation if a member of
          `stop_sequences` is generated
        temperature (float): Sampling temperature for logits sampling
        top_k (int): Number of highest probability vocabulary tokens to keep for top-k
            sampling
        top_p (float): If <1, only the most probable tokens with probabilities that add
            up to `top_p` or higher are kept for top-p sampling
        truncate (int): Truncate input to this length if set
        typical_p (float): Typical decoding mass.
        watermark (bool): Whether to add a watermark to the generated text
            Defaults to False.
        decoder_input_details (bool): Whether to return the decoder input token
            logprobs and ids. Requires `details` to be set to True as well.
            Defaults to False.

    """

    prompt: str
    details: bool = Field(default=False)
    stream: bool = Field(default=False)
    do_sample: bool = Field(default=False)
    max_new_tokens: int = Field(default=20)
    best_of: Optional[int] = None
    repetition_penalty: Optional[float] = None
    return_full_text: bool = Field(default=False)
    seed: Optional[int] = None
    stop_sequences: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    truncate: Optional[int] = None
    typical_p: Optional[float] = None
    watermark: bool = Field(default=False)
    decoder_input_details: bool = Field(default=False)


class HFSummarizationConfig(ConfigDict):
    """Summarization model configuration

    Args:
        model (str): Model name
        max_length (int): Maximum length in tokens of the generated summary
        min_length (int): Minimum length in tokens of the generated summary
        top_k (int): Number of top tokens to sample from
        top_p (float): Cumulative probability for top-k sampling
        temperature (float): Temperature for sampling. Default 1.0
        repetition_penalty (float): Repetition penalty for beam search
        num_return_sequences (int): Number of sequences to return
        use_cache (bool): Whether to use cache during inference
    """

    max_length: NotRequired[int]
    min_length: NotRequired[int]
    top_k: NotRequired[int]
    top_p: NotRequired[float]
    temperature: NotRequired[float]
    repetition_penalty: NotRequired[float]
    max_time: NotRequired[float]


class HFSummarizationInferenceInput(HFInferenceInput):
    """Input data for summarization models

    Args:
        text (str): Text to summarize
        parameters (Optional[HFSummarizationConfig]): Summarization model
    """

    text: str
    parameters: Optional[dict[str, Any]] = None


class HFClassificationInferenceOutput(BaseModel):
    """Output data for HF classification models

    Args:
        label (str): Predicted label
        score (float): Confidence score for the predicted label
    """

    label: str
    score: float

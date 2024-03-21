"""
Module containing data models used by the service
"""
from enum import IntEnum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, StringConstraints, model_validator

HexStr = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern="^[a-fA-F0-9]+$")
]


class InfernetInputSource(IntEnum):
    CHAIN = 0
    OFFCHAIN = 1


class InfernetInput(BaseModel):
    """
    Infernet containers must accept InfernetInput. Depending on the source (onchain vs.
     offchain), the associated data object is either a hex string from an onchain
    source meant to be decoded directly, or a data dictionary (off chain source).
    """

    source: InfernetInputSource
    data: Union[HexStr, dict[str, Any]]

    @model_validator(mode="after")
    def check_data_correct(self) -> "InfernetInput":
        src = self.source
        dta = self.data
        if (
            src is not None
            and dta is not None
            and (
                (src == InfernetInputSource.CHAIN and not isinstance(dta, str))
                or (src == InfernetInputSource.OFFCHAIN and not isinstance(dta, dict))
            )
        ):
            raise ValueError(
                f"InfernetInput data type ({type(dta)}) incorrect for source ({str(src)})"  # noqa: E501
            )
        return self


class ConvoMessage(BaseModel):
    """
    A convo message is a part of a conversation.
    """

    role: str  # who the content is attributed to
    content: str  # actual content of the convo message


class CSSCompletionParams(BaseModel):
    """
    A CSS Completion param has a list of Convo message.
    """

    endpoint: Literal["completions"]
    messages: list[ConvoMessage]


class CSSEmbeddingParams(BaseModel):
    """
    A CSS Embeddng Param has an input string param.
    """

    endpoint: Literal["embeddings"]
    input: str


class CSSRequest(BaseModel):
    """A CSSRequest, meant for querying closed source models."""

    # name of model to use. Valid values depends on the the CSS model provider
    model: str
    # parameters associated with the request. Can either be a Completion
    # or an Embedding Request
    params: Union[CSSCompletionParams, CSSEmbeddingParams] = Field(
        ..., discriminator="endpoint"
    )

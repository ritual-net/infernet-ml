"""
Model Loader: depending on the model source, load the model from the local file system,
Hugging Face Hub, or Arweave.
"""

import logging
from enum import Enum
from typing import Union, cast

from huggingface_hub import hf_hub_download  # type: ignore


class ModelSource(Enum):
    """
    Enum for the model source
    """

    LOCAL = "local"
    ARWEAVE = "arweave"
    HUGGINGFACE_HUB = "huggingface_hub"


logger: logging.Logger = logging.getLogger(__name__)


def load_model(model_source: ModelSource, **kwargs: Union[str, list[str]]) -> str:
    """
    Load the model from the specified source.

    Args:
        model_source (ModelSource): the source of the model
        **kwargs (str): the arguments for the model source

    Returns:
        str: the path to the model
    """

    # repo_id & filename are common for both Hugging Face Hub and Arweave
    def _get_info() -> tuple[str, str]:
        return cast(str, kwargs["repo_id"]), cast(str, kwargs["filename"])

    match model_source:
        # load the model locally
        case ModelSource.LOCAL:
            model_path: str = cast(str, kwargs["model_path"])
            logging.info(f"Loading model from local path {model_path}")
            return model_path
        case ModelSource.HUGGINGFACE_HUB:
            (repo_id, filename) = _get_info()
            logging.info(
                f"Downloading model from Hugging Face Hub {repo_id}"
                f" with filename {filename}"
            )
            return cast(str, hf_hub_download(repo_id, filename))
        case ModelSource.ARWEAVE:
            (repo_id, filename) = _get_info()
            owners: list[str] = cast(list[str], kwargs["owners"])
            logging.info(
                f"Downloading model from Arweave {repo_id} with filename {filename}"
            )
            try:
                from infernet_ml.utils.arweave import download_model_file

                return download_model_file(
                    model_id=repo_id,
                    model_file_name=filename,
                    owners=owners,
                )
            except ImportError as e:
                logger.error(
                    f"Arweave is not installed: {e} please install it by "
                    f'running `pip install "pyarweave @ git+https://github.com/ritual-net/pyarweave.git"`'
                )
                raise
        case _:
            raise ValueError(f"Invalid model source {model_source}")

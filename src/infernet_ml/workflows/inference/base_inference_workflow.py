"""
Module implementing a base inference workflow.

This class is not meant to be subclassed directly; instead,
subclass one of [TGIClientInferenceWorkflow, CSSInferenceWorkflow,
 BaseClassicInferenceWorkflow]


"""

import abc
import logging
from typing import Any, final

logger: logging.Logger = logging.getLogger(__name__)


class BaseInferenceWorkflow(metaclass=abc.ABCMeta):
    """
    Base class for an inference workflow
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Constructor. keeps track of arguments passed in.
        """
        super().__init__()
        self.args: list[Any] = list(args)
        self.kwargs: dict[Any, Any] = kwargs

        self.is_setup = False
        self.__inference_count: int = 0
        self.__proof_count: int = 0

    def setup(self) -> Any:
        """
        calls setup and keeps track of whether or not setup was called
        """
        self.is_setup = True
        return self.do_setup()

    @abc.abstractmethod
    def do_setup(self) -> Any:
        """set up your workflow here.
        For LLMs, this may be parameters like top_k, temperature
        for classical LLM, this may be model hyperparams.

        Returns: Any
        """

    @final
    def inference(self, input_data: Any) -> Any:
        """performs inference. Checks that model is set up before
        performing inference.
        Subclasses should implement do_inference.

        Args:
            input_data (typing.Any): input from user

        Raises:
            ValueError: if setup not called beforehand

        Returns:
            Any: result of inference
        """
        if not self.is_setup:
            raise ValueError("setup not called before inference")

        logging.info("preprocessing input_data %s", input_data)
        preprocessed_data = self.do_preprocessing(input_data)

        logging.info("querying model with %s", preprocessed_data)
        model_output = self.do_run_model(preprocessed_data)

        logging.info("postprocessing model_output %s", model_output)
        self.__inference_count += 1
        return self.do_postprocessing(input_data, model_output)

    @abc.abstractmethod
    def do_run_model(self, preprocessed_data: Any) -> Any:
        """run model here. preprocessed_data type is
            left generic since depending on model type
        Args:
            preprocessed_data (typing.Any): preprocessed input into model

        Returns:
            typing.Any: result of running model
        """
        pass

    @abc.abstractmethod
    def do_preprocessing(self, input_data: Any) -> Any:
        """
        Implement any preprocessing of the raw user input.
        For example, you may need to apply feature engineering
        on the input before it is suitable for model inference.

        Args:
            data: raw user input

        Returns:
            str: transformed input
        """
        return input_data

    @abc.abstractmethod
    def do_postprocessing(self, input_data: Any, output_data: Any) -> Any:
        """
        Implement any postprocessing here. for ease of
        serving, we must return a dict or string.

        Args:
            output_data (Any):  raw output from model

        Returns:
            typing.Any: result of postprocessing
        """

    @final
    def generate_proof(self) -> None:
        """
        Generates proof. checks that setup performed before hand.
        """
        if self.__inference_count <= self.__proof_count:
            logging.warning(
                "generated %s inferences only but "
                + "already generated %s. Possibly duplicate proof.",
                self.__inference_count,
                self.__proof_count,
            )

        self.do_generate_proof()
        self.__proof_count += 1

    def do_generate_proof(self) -> Any:
        """
        Generates proof, which may vary based on proving system. We currently
        only support EZKL based proving, which does not require proof
        generation to be defined as part of the inference workflow if
        the inference from the circuit directly used for proving. Indeed,
        that is the case for the classic infernet_ml proof service. However, this
        may change with the usage of optimistic and other eager or lazy
        proof systems, which we intend to support in the future. By default,
        will raise NotImplementedError. Override in subclass as needed.
        """
        raise NotImplementedError

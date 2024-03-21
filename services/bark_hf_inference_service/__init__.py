"""
this module serves as the driver for the tts inference service.
"""
import logging
import os
import pathlib
from typing import Any, Optional, Union, cast

from quart import Quart, Response, request, send_file
from scipy.io.wavfile import write as write_wav  # type: ignore

from infernet_ml.utils.arweave import load_wallet, upload
from infernet_ml.utils.service_models import InfernetInput, InfernetInputSource
from infernet_ml.workflows.inference.bark_hf_inference_workflow import (
    BarkHFInferenceWorkflow,
    BarkWorkflowInput,
)

logger: logging.Logger = logging.getLogger(__name__)


def env_strip_quotes(var_name: str, default: str) -> str:
    val = os.getenv(var_name, default)
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        val = val[1:-1]
    return val


def create_app(test_config: Optional[dict[str, Any]] = None) -> Quart:
    """
    Create a new Quart app and set up the TTS (text-to-speech) inference service.
    Returns:
        Quart: TTS inference service app
    """

    app_config = test_config or {
        "workflow_kw": {
            "model_name": env_strip_quotes("MODEL_NAME", "suno/bark"),
            "default_voice_preset": env_strip_quotes(
                "DEFAULT_VOICE_PRESET", "v2/en_speaker_6"
            ),
        },
        "upload_to_arweave": "true"
        in env_strip_quotes("UPLOAD_TO_ARWEAVE", "False").lower(),
    }

    workflow_kw_args = app_config["workflow_kw"]
    upload_to_arweave = app_config["upload_to_arweave"]

    if upload_to_arweave:
        # ensure arweave wallet exists.
        load_wallet()

    workflow = BarkHFInferenceWorkflow(**workflow_kw_args)
    app = Quart(__name__)

    # path to the generated audio file, will get overridden with the next request
    generated_audio_path = "/root/audio/output.wav"

    # download the model
    workflow.setup()

    @app.route("/")
    async def index() -> str:
        return "BARK service!"

    @app.route("/get_audio", methods=["GET"])
    async def get_audio() -> Response:
        """
        Get the last generated audio file. For testing purposes.

        Returns:
            Response: the generated audio file
        """
        return await send_file(
            generated_audio_path,
            mimetype="audio/wav",
            as_attachment=True,
            attachment_filename="audio.wav",
        )

    @app.route("/service_output", methods=["POST"])
    async def inference() -> Union[Response, dict[str, str]]:
        # get the input data
        _req_body = await request.get_json()
        print(f"req body: {_req_body}")
        _input: InfernetInput = InfernetInput(**_req_body)

        if _input.source == InfernetInputSource.OFFCHAIN:
            inference_input = BarkWorkflowInput(**cast(dict[str, Any], _input.data))
        else:
            raise NotImplementedError("on-chain tts inference is not supported yet.")

        # get the prompt and generate the audio
        inference_result = workflow.inference(inference_input)

        write_wav(
            generated_audio_path,
            BarkHFInferenceWorkflow.SAMPLE_RATE,
            inference_result.audio_array,
        )

        response = {"output": "success"}

        if upload_to_arweave:
            tx = upload(
                pathlib.Path(generated_audio_path),
                {
                    "Content-Type": "audio/wav",
                    "File-Name": os.path.basename(generated_audio_path),
                },
            )
            response["output"] = tx.id

        return response

    return app

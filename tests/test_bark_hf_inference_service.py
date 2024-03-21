import os
from unittest.mock import MagicMock

import pytest
from torch import Tensor
from transformers import BatchEncoding  # type: ignore

arweave_hash = "mock-areweave-hash"
arweave_keyfile_path = "some-path/keyfile-arweave.json"


@pytest.fixture()
def app_and_mocks(mocker):  # type: ignore
    bark_loader: MagicMock = mocker.patch(
        "transformers.BarkModel.from_pretrained", return_value=mocker.MagicMock()
    )
    auto_processor_loader: MagicMock = mocker.patch(
        "transformers.AutoProcessor.from_pretrained", return_value=mocker.MagicMock()
    )

    write_wav_mock: MagicMock = mocker.patch("scipy.io.wavfile.write")

    upload_mock: MagicMock = mocker.patch("infernet_ml.utils.arweave.upload")
    upload_mock.return_value.id = arweave_hash

    orig_getenv = os.getenv
    mocker.patch(
        "os.getenv",
        side_effect=lambda key: arweave_keyfile_path
        if key == "ARWEAVE_WALLET_FILE_PATH"
        else orig_getenv(key),
    )
    path_exists = mocker.patch(
        "os.path.exists",
        side_effect=lambda path: path == arweave_keyfile_path,
    )
    mock_wallet = mocker.patch("ar.Wallet.__init__", return_value=None)

    from services.bark_hf_inference_service import create_app

    app = create_app(
        {
            "workflow_kw": {
                "model_name": "suno/bark",
                "default_voice_preset": "v2/en_speaker_5",
            },
            "upload_to_arweave": True,
        }
    )

    bark_loader.assert_called_once_with("suno/bark")
    bark_loader.return_value.to.assert_called_once_with("cpu")
    auto_processor_loader.assert_called_once_with("suno/bark")
    path_exists.assert_called_with(arweave_keyfile_path)
    mock_wallet.assert_called_once_with(arweave_keyfile_path)
    bark_model = bark_loader.return_value.to.return_value
    bark_processor = auto_processor_loader.return_value

    # other setup can go here

    yield app, bark_model, bark_processor, write_wav_mock, upload_mock

    # clean up / reset resources here


@pytest.fixture()
def client(app_and_mocks):  # type: ignore
    app, _, _, _, _ = app_and_mocks
    return app.test_client()


@pytest.fixture()
def runner(app_and_mocks):  # type: ignore
    app, _, _, _, _ = app_and_mocks
    return app.test_cli_runner()


@pytest.mark.asyncio
async def test_hello(client, app_and_mocks, mocker):  # type: ignore
    res = await client.get("/")
    assert res.status_code == 200
    res_text = await res.get_data(as_text=True)
    assert res_text == "BARK service!"


@pytest.mark.asyncio
async def test_generate_audio(client, app_and_mocks):  # type: ignore
    app, bark_model, bark_processor, write_wav, upload = app_and_mocks
    bark_model.generate.return_value = Tensor([1, 2, 3])
    bark_processor.return_value.to.return_value = BatchEncoding({"some": "data"})

    res = await client.post(
        "/service_output",
        json={
            "source": 1,
            "data": {
                "prompt": "Hello, world!",
                "voice_preset": "v2/en_speaker_5",
            },
        },
    )

    assert res.status_code == 200
    res_dict = await res.get_json()
    assert res_dict.get("output") == arweave_hash

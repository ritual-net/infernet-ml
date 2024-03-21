import json
import os

import pytest
from eth_abi import decode  # type: ignore
from eth_abi import encode  # type: ignore

from services.hf_inference_client_service import create_app


@pytest.fixture()
def app(mocker):  # type: ignore
    app = create_app(
        "text_classification",
        test_config={"HF_TOKEN": os.environ.get("HF_TOKEN")},
    )
    yield app


@pytest.fixture()
def client(app):  # type: ignore
    return app.test_client()


@pytest.fixture()
def runner(app):  # type: ignore
    return app.test_cli_runner()


@pytest.mark.asyncio
async def test_inference_offchain(client, mocker) -> None:  # type: ignore
    res = await client.get("/")
    assert res.status_code == 200
    res = await client.post(
        "/service_output", json={"source": 1, "data": {"text": "Hello World!"}}
    )
    assert res.status_code == 200
    output = await res.get_json()
    output = output["output"]
    assert output[0]["label"] == "POSITIVE"
    assert output[0]["score"] > 0.6


@pytest.mark.asyncio
async def test_inference_onchain(client, mocker) -> None:  # type: ignore
    res = await client.get("/")
    assert res.status_code == 200

    data = {"text": "Hello World!"}
    # Encode the data into bytes
    data_json = json.dumps(data).encode("utf-8")
    # Define the ABI type strings
    abi_types = ["bytes[]"]
    encoded_data = encode(abi_types, [[data_json]])
    res = await client.post(
        "/service_output",
        json={
            "source": 0,
            "data": encoded_data.hex(),
        },
    )
    assert res.status_code == 200
    output = await res.get_data()
    # Decode the result from bytes[] to json
    output = bytes.fromhex(output.decode("utf-8"))
    abi_types = ["bytes[]"]
    output = decode(abi_types, output)
    output = json.loads(output[0][0].decode("utf-8"))

    output = output["output"]
    assert output[0]["label"] == "POSITIVE"
    assert output[0]["score"] > 0.6

import os
from typing import Callable

import numpy as np
import pytest
from eth_abi import encode  # type: ignore
from quart.typing import TestClientProtocol

from infernet_ml.utils.model_loader import ModelSource
from services.onnx_inference_service import create_app

FLOAT_DECIMALS = 9

hf_args = {
    "kwargs": {
        "model_source": ModelSource.HUGGINGFACE_HUB,
        "model_args": {
            "repo_id": "Ritual-Net/iris-classification",
            "filename": "iris.onnx",
        },
    }
}

arweave_args = {
    "kwargs": {
        "model_source": ModelSource.ARWEAVE,
        "model_args": {
            "repo_id": "Ritual-Net/iris-classification",
            "filename": "iris.onnx",
            "owners": [os.environ.get("MODEL_OWNER")],
        },
    }
}

AppConfig = dict[str, str]
ClientFactory = Callable[[dict[str, str]], TestClientProtocol]


@pytest.fixture()
def client_factory() -> ClientFactory:
    def _factory(args: AppConfig) -> TestClientProtocol:
        app = create_app(args)
        return app.test_client()

    return _factory


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [hf_args, arweave_args])
async def test_hf(client_factory: ClientFactory, args: AppConfig) -> None:
    res = await client_factory(args).get("/")
    assert res.status_code == 200
    data = await res.get_data(as_text=True)
    assert data == "ONNX Inference Service!"


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [hf_args, arweave_args])
async def test_inference_offchain(
    client_factory: ClientFactory, args: AppConfig
) -> None:
    res = await client_factory(args).post(
        "/service_output",
        json={
            "source": 1,
            "data": {"input": [[1.0380048, 0.5586108, 1.1037828, 1.712096]]},
        },
    )
    assert res.status_code == 200
    predictions = await res.get_json()
    result = np.array(predictions[0]).argmax(axis=1)
    assert result == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [hf_args, arweave_args])
async def test_inference_onchain(
    client_factory: ClientFactory, args: AppConfig
) -> None:
    inputs = [
        int(a * 10**FLOAT_DECIMALS)
        for a in [1.0380048, 0.5586108, 1.1037828, 1.712096]
    ]
    # encode web3 abi.encode(uint64, uint64, uint64, uint64)
    data = encode(["uint64", "uint64", "uint64", "uint64"], inputs).hex()

    res = await client_factory(args).post(
        "/service_output",
        json={
            "source": 0,
            "data": data,
        },
    )
    assert res.status_code == 200
    predictions = await res.get_json()
    result = np.array(predictions[0]).argmax(axis=1)
    assert result == 2

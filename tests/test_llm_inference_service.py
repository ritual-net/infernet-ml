import logging
from collections import namedtuple
from typing import Any
from unittest.mock import DEFAULT

import pytest
from text_generation.errors import ShardNotReadyError  # type: ignore

from services.llm_inference_service import create_app

MockContainer = namedtuple("MockContainer", ["generated_text"])


class MockClient:
    generated_text: str = "for sure"

    @staticmethod
    def update_ret_val(val: str) -> Any:
        if val == "ShardNotReadyError":
            logging.info("Raising ShardNotReadyError")
            raise ShardNotReadyError("ERROR")
        logging.info("client queried with %s", val)
        return DEFAULT


@pytest.fixture()
def app(mocker):  # type: ignore
    mocker.patch(
        "text_generation.Client.generate",
        return_value=MockClient,
        side_effect=MockClient.update_ret_val,
    )

    app = create_app(test_config={})
    app.config.update(
        {
            "TESTING": True,
        }
    )

    # other setup can go here

    yield app

    # clean up / reset resources here


@pytest.fixture()
def client(app):  # type: ignore
    return app.test_client()


@pytest.fixture()
def runner(app):  # type: ignore
    return app.test_cli_runner()


@pytest.mark.asyncio
async def test_inference(client, mocker):  # type: ignore
    res = await client.post("/inference", json={"key": "152435555"})
    assert res.status_code == 400, await res.get_json()

    res = await client.post("/inference", json={"key": "1552342424", "messageId": 1234})
    assert res.status_code == 400, await res.get_json()

    res = await client.post(
        "/inference",
        json={
            "source": 1,
            "data": {"text": "wagmi"},
        },
    )
    assert await res.get_data(as_text=True) == "for sure", await res.get_data(
        as_text=True
    )

    res = await client.post(
        "/service_output",
        json={
            "source": 1,
            "data": {
                "text": "ShardNotReadyError",
            },
        },
    )
    assert await res.get_json() == {
        "code": "500",
        "name": "Internal Server Error",
        "description": "ERROR",
    }

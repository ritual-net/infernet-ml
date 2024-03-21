import logging
import os
from pathlib import Path

from click.testing import CliRunner
from pytest import LogCaptureFixture

from infernet_ml.utils import arweave
from infernet_ml.utils.arweave import cli

OWNER = os.environ.get("MODEL_OWNER", "")


def test_download_model(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        args = [
            "download-model",
            "--model_id",
            "test_upload",
            "--owner",
            OWNER,
        ]
        result = runner.invoke(cli, args)

        logging.warning(result.output)

        if result.exception:
            logging.exception(result.exception)

        assert not result.exception
        assert result.exit_code == 0


def test_download_file(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    test_file: str = arweave.download_model_file(
        "test_upload",
        "test_file.txt",
        owners=[OWNER],
        base_path=str(tmp_path),
    )

    assert os.path.exists(test_file)

"""
Script containing a command line interface to upload and download model files
from arweave.

Model files are logically grouped together via a Manifest file, which maps individual
transaction data to named files.

When uploading a model directory, a version mapping dictionary file is expected to be
provided. The mapping should contain a map of filename to version tag. The version tag
is useful if a specific version of a file is meant to be downloaded. If no mapping is
specified, the empty string is used by default.
"""

import hashlib
import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Optional

import click
from ar import Peer  # type: ignore
from ar import Transaction, Wallet
from ar.manifest import Manifest  # type: ignore
from ar.utils.serialization import b64dec  # type: ignore
from ar.utils.transaction_uploader import get_uploader  # type: ignore
from tqdm import tqdm

# Old gateways support 10 MB,
# new ones support 12 MB,
# we take lower to be conservative
MAX_NODE_BYTES = 1e7

logger = logging.getLogger(__name__)


def get_tags_dict(tag_dicts: list[dict[str, str]]) -> dict[str, str]:
    """
    Helper function to merge a list of tag dicts into
    a single dictionary.

    Args:
        tag_dicts(list[dict[str, str]): a list of tag dicts with
        keys 'name' and 'value' corresponding to the name and
        value of the tag respectively.

    Returns:
        dict[str, str]: a key value dict mapping tag name to tag value
    """
    tags: dict[str, str] = {item["name"]: item["value"] for item in tag_dicts}
    return tags


def edge_unix_ts(edge: dict[str, Any]) -> float:
    """
    Helper function to extract the unix time stamp from an
    Arweave transaction edge. See https://arweave.net/graphql for the
    Arweave graphql schema.

    Args:
        edge (dict[str, Any]): a transaction edge object

    Returns:
        float: unix timestamp in seconds
    """
    # sort matching manifests by time, get latest
    tag_dicts: list[dict[str, str]] = edge["node"]["tags"]
    return float(get_tags_dict(tag_dicts)["Unix-Time"])


def get_sha256_digest(file_path: str) -> str:
    """Helper function that computes the digest
    of a file in binary mode to handle potentially
    large files.

    Args:
        file_path (str): path to a file

    Returns:
        str: hex string representing the sha256
    """
    h = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


@click.group()
def cli() -> None:
    pass


@click.option(
    "--model_id",
    type=str,
    prompt="enter model_id",
)
@click.option(
    "--path_str",
    type=click.Path(exists=True, readable=True),
    prompt="enter model directory",
)
@click.option(
    "--version_mapping_json_path",
    type=str,
    prompt="enter version mapping json of model files to upload",  # noqa: E501
)
@cli.command()
def upload_model(
    model_id: str,
    path_str: str,
    version_mapping_json_path: str,
) -> str:
    """
    Uploads a model directory to Arweave. For every model upload, a manifest mappping
    is created. Please set the ARWEAVE_WALLET_FILE_PATH environment variable before
    using.

    Args:
        model_id (str): id associated with the model. Generally, this looks like
            MODEL_ORG/MODEL_NAME
        path_str (str): directory path
        version_mapping_json_path (str): path to a json dict file mapping file names to
            specific versions. If a specific mapping is found, the File-Version
            attribute is tagged with the value. This is to faciliate uploading and
            downloading version specific files.

    Raises:
        ValueError: if ARWEAVE_WALLET_FILE_PATH not set

    Returns:
        str: url to the manifest file
    """
    wallet = load_wallet()

    # path to load files from
    path: Path = Path(path_str)

    # load all sub-paths in this path
    p = path.glob("**/*")

    # get timestamp to tag files with
    timestamp = time.time()

    # filter out simlinks and non-files
    files = [x for x in p if x.is_file()]

    with open(version_mapping_json_path, "r") as version_mapping_file:
        version_mapping = json.load(version_mapping_file)
        click.echo(f"using mapping {version_mapping}")

    # keep track of entries via a manifest
    manifest_dict: dict[str, str] = {}

    for f in files:
        rel_path = os.path.relpath(f, path)
        click.echo(f"looking at {f} ({rel_path}) Size: {os.path.getsize(f)}")

        tags_dict = {
            "Content-Type": guess
            if (guess := mimetypes.guess_type(f)[0])
            else "application/octet-stream",
            "App-Name": "Ritual",
            "App-Version": "0.0.1",
            "Unix-Time": str(timestamp),
            "Model-Id": str(model_id),
            "File-Version": version_mapping.get(str(rel_path), ""),
            "File-Name": rel_path,
            "File-SHA256": get_sha256_digest(str(f)),
        }

        tx = upload(f, tags_dict)

        # we are done uploading the whole file, keep track if filename -> tx.id
        manifest_dict[str(os.path.relpath(f, path_str))] = tx.id
        click.echo(f"uploaded file {f} with id {tx.id} and tags {tags_dict}")

    # we create a manifest of all the files to their transactions
    m = Manifest(manifest_dict)

    # upload the manifest
    t = Transaction(wallet, data=m.tobytes())

    t.add_tag("Content-Type", "application/x.arweave-manifest+json")
    t.add_tag("Type", "manifest")
    t.add_tag("App-Name", "Ritual")
    t.add_tag("App-Version", "0.0.1")
    t.add_tag("Unix-Time", str(timestamp))
    t.add_tag("Model-Id", str(model_id))

    t.sign()
    t.send()

    click.echo(f"uploaded manifest with id {t.id}")

    return f"{t.api_url}/{t.id}"


def file_exists(file_path: str, txid: str) -> bool:
    query_str = (
        """
            query {
                transaction(
                id: "%s"
                )
                {
                    owner{
                        address
                    }
                    data{
                        size
                        type
                    }
                    tags{
                        name
                        value
                    }
                }
            }
        """
        % txid
    )

    res = Peer().graphql(query_str)

    tx_file_size: int = int(res["data"]["transaction"]["data"]["size"])
    tx_tags: dict[str, str] = get_tags_dict(res["data"]["transaction"]["tags"])

    local_file_exists: bool = os.path.exists(file_path)

    size_matches: bool = (
        tx_file_size == os.path.getsize(file_path) if local_file_exists else False
    )

    digest_matches: bool = (
        tx_tags.get("File-SHA256") == get_sha256_digest(file_path)
        if local_file_exists
        else False
    )

    logger.info(
        "file_path=%s local_file_exists=%s size_matches=%s digest_matches=%s",
        file_path,
        local_file_exists,
        size_matches,
        digest_matches,
    )

    return local_file_exists and size_matches and digest_matches


def load_wallet() -> Wallet:
    """
    Helper function to load the wallet from the ARWEAVE_WALLET_FILE_PATH environment
    variable.
    :return: Wallet object
    """
    if not (wallet_file_path := os.getenv("ARWEAVE_WALLET_FILE_PATH")):
        raise ValueError("ARWEAVE_WALLET_FILE_PATH environment variable not set")

    if not os.path.exists(wallet_file_path):
        raise ValueError(f"Wallet file {wallet_file_path} does not exist.")

    # wallet used to pay for file upload
    return Wallet(wallet_file_path)


def upload(file_path: Path, tags_dict: dict[str, str]) -> Transaction:
    wallet = load_wallet()
    with open(file_path, "rb", buffering=0) as file_handler:
        tx = Transaction(wallet, file_handler=file_handler, file_path=file_path)

        for n, v in tags_dict.items():
            tx.add_tag(n, v)
        tx.sign()

        # uploader required to upload in chunks
        uploader = get_uploader(tx, file_handler)
        # manually update tqdm progress bar to total chunks
        with tqdm(total=uploader.total_chunks) as pbar:
            while not uploader.is_complete:
                # upload a chunk
                uploader.upload_chunk()
                # increment progress bar by 1 chunk
                pbar.update(1)

    return tx


def download(pathname: str, txid: str) -> str:
    """function to dowload an arweave data tx to a given path

    Args:
        pathname (str): path to download to
        txid (str): txid of the data transaction

    Returns:
        str: absolute path of the downloaded file
    """
    p = Peer()

    loaded_bytes = 0

    with open(pathname, "wb") as binary_file:
        try:
            # try downloading the transaction data directly
            # from the default data endpoint
            data = p.data(txid)

            # write downloaded file to disk
            binary_file.write(data)

            return os.path.abspath(pathname)
        except Exception:
            logger.exception(
                f"failed to download transaction data for {txid} directly."
                + " Will try downloading in chunks."
            )

        """
         if we are unable to download files directly, likely the file is too big.
         we can download in chunks.
         To do so, start with the end offset and fetch a chunk.
         Subtract its size from the transaction size.
         If there are more chunks to fetch, subtract the size of the chunk
         from the offset and fetch the next chunk.
         Note that chunks seem to take some time to be
         available even after a transaction may be finalized.
         For more information see:
         https://docs.arweave.org/developers/arweave-node-server/http-api
        """

        chunk_offset: dict[str, int] = p.tx_offset(txid)
        size: int = chunk_offset["size"]
        startOffset: int = chunk_offset["offset"] - size + 1

        if size < MAX_NODE_BYTES:
            # if the size is less than the maximum node download size
            # just download the file to disk via the tx_data endpoint
            # which purportedly downloads files regardness of how it
            # was uploaded (but has this size limitation)
            data = p.tx_data(txid)
            binary_file.write(data)
        else:
            with tqdm(total=size) as pbar:
                while loaded_bytes < size:
                    # download this chunk
                    chunkData = p.chunk(startOffset + loaded_bytes)["chunk"]
                    # arweave files use b64 encoding. We decode the chunk here
                    chunkDataDec = b64dec(chunkData)
                    # write the part of the file to disk
                    binary_file.write(chunkDataDec)
                    # update offset to subtract from file size
                    loaded_bytes += len(chunkDataDec)
                    # update progress bar
                    pbar.update(len(chunkDataDec))

        return os.path.abspath(pathname)


@click.option(
    "--model_id",
    type=str,
    prompt="enter model_id",
)
@click.option(
    "--owner",
    type=str,
    default=[],
    multiple=True,
)
@click.option("--force_download", is_flag=True)
@click.option("--base_path", type=str, default="")
@cli.command()
def download_model(
    model_id: str,
    owner: list[str] = [],
    base_path: str = "",
    force_download: bool = False,
) -> list[str]:
    """Downloads a model from Arweave to a given directory.

    Args:
        model_id (str): id of model
        owner (list[str]): list of owners for the given model. If empty list provided,
            defaults to address of ARWEAVE_WALLET_FILE_PATH.
        base_path (str, optional): Directory to download to. Defaults to current
            directory.

    Raises:
        ValueError: if ARWEAVE_WALLET_FILE_PATH not specified
        ValueError: if matching model manifest not found

    Returns:
        list[str]: downloaded file paths
    """
    peer = Peer()
    base = Path(base_path)
    if not Path.exists(base):
        os.makedirs(base)

    if len(owner) == 0:
        # default to current wallet address
        if not (wallet_file_path := os.getenv("ARWEAVE_WALLET_FILE_PATH")):
            raise ValueError("ARWEAVE_WALLET_FILE_PATH environment variable not set")

        owner = [Wallet(wallet_file_path).address]

    query_str = """
    query {
        transactions(
            sort:HEIGHT_DESC,
            owners: %s,
            tags: [
                {
                    name: "App-Name",
                    values: ["Ritual"]
                },
                {
                    name: "Model-Id",
                    values: ["%s"]
                },
                {
                    name: "Type",
                    values: ["manifest"]
                }
            ]
        )
        {
            edges {
                node {
                    block {
                        id
                        timestamp
                    }
                    id
                    owner {
                        address
                    }
                    tags {
                        name
                        value
                    }
                }
            }
        }
    }
    """ % (
        json.dumps(owner),
        model_id,
    )

    click.echo(query_str)
    res = peer.graphql(query_str)

    # get latest Manifest

    # sort matching manifests by time, get latest
    res["data"]["transactions"]["edges"].sort(reverse=True, key=edge_unix_ts)

    if len(res["data"]["transactions"]["edges"]) == 0:
        raise ValueError("Could not find any matching model manifests from query")

    tx_id = res["data"]["transactions"]["edges"][0]["node"]["id"]

    # download manifest data
    click.echo(f"found manifest {res['data']['transactions']['edges'][0]['node']}")

    m = json.loads(peer.tx_data(tx_id))

    click.echo(f"loaded manifest {m}")

    paths = []
    # download files in manifest
    for pathname, tid in m["paths"].items():
        file_tid: str = tid["id"]
        joined_path: Path = base.joinpath(pathname)

        # check if file exists
        if force_download or not file_exists(str(joined_path), file_tid):
            st = time.time()
            click.echo(f"downloading file {pathname} for {file_tid}")
            paths.append(download(str(joined_path), file_tid))
            click.echo(f"downloaded in {time.time() - st} sec: {joined_path}")
        else:
            click.echo(
                f"Path {joined_path} already exists and will not be downloaded. "
                + "Please remove it or use --force_download flag."
            )

    return paths


def download_model_file(
    model_id: str,
    model_file_name: str,
    file_version: Optional[str] = None,
    owners: Optional[list[str]] = None,
    force_download: bool = False,
    base_path: str = "",
) -> str:
    """Downloads a specific model file from Arweave.

    Args:
        model_id (str): model id
        model_file_name (str): name of model file
        file_version (Optional[str], optional): Version of file. Defaults to None.
        owners (list[str], optional): List of owners allowed for file. If None
        specified, will default to owner address for ARWEAVE_WALLET_FILE_PATH
        environment variable.
        base_path (str, optional): path to download file to. Defaults to "".

    Raises:
        ValueError: If ARWEAVE_WALLET_FILE_PATH not specified

    Returns:
        str: path of downloaded file
    """
    base = Path(base_path)
    if not Path.exists(base):
        os.makedirs(base)

    if not owners:
        # default to current wallet address
        if not (wallet_file_path := os.getenv("ARWEAVE_WALLET_FILE_PATH")):
            raise ValueError("ARWEAVE_WALLET_FILE_PATH environment variable not set")

        owners = [Wallet(wallet_file_path).address]

    file_version_str = (
        ""
        if not file_version
        else """
        {
            name: "File-Version",
            values: ["%s"]
        },
    """
    )
    query_str = """
    query {
        transactions(
            sort:HEIGHT_DESC,
            owners: %s,
            tags: [
                {
                    name: "App-Name",
                    values: ["Ritual"]
                },
                %s
                {
                    name: "File-Name",
                    values: ["%s"]
                },
                {
                    name: "Model-Id",
                    values: ["%s"]
                }
            ])
        {
            edges {
                node {
                    block {
                        id
                        timestamp
                    }
                    id
                    owner {
                        address
                    }
                    tags {
                        name
                        value
                    }
                }
            }
        }
    }
    """ % (
        json.dumps(owners),
        file_version_str,
        model_file_name,
        model_id,
    )
    logger.debug(query_str)

    file_path: Path = base.joinpath(model_file_name)

    res = Peer().graphql(query_str)

    res["data"]["transactions"]["edges"].sort(reverse=True, key=edge_unix_ts)

    tx_metadata: dict[str, Any] = res["data"]["transactions"]["edges"][0]["node"]

    tx_id = tx_metadata["id"]

    if force_download or not file_exists(str(file_path), tx_id):
        logger.info(f"downloading {tx_metadata}")
        return download(str(file_path), tx_id)

    else:
        logger.info(f"not downloading {tx_metadata} because it already exists")
        return os.path.abspath(file_path)


if __name__ == "__main__":
    cli()  # pylint: disable=E1120:no-value-for-parameter

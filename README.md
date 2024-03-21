# infernet-ml

`infernet-ml` is a lightweight library meant to simplify the implementation
of machine learning workflows for models intended for Web3.

# Installation

### Via `pip`

To install this library via pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "infernet-ml"
```

### Via `uv`

Alternatively, via [uv](https://github.com/astral-sh/uv):

```bash
uv venv
source .venv/bin/activate
uv pip install infernet-ml
```

## Optional Dependencies

Depending on the workflow you're using, you may want to install optional dependencies. For example, if you're using the
`torch` workflow, you'll need to install its dependencies by running:

```bash
pip install "infernet-ml[torch_inference]"
```

Alternatively, via [uv](https://github.com/astral-sh/uv):

```bash
uv pip install "infernet-ml[torch_inference]"
```

> [!NOTE] The optional dependencies for this workflow require that `cmake` is installed on your system. You can install
`cmake` on MacOS by running `brew install cmake`. On Ubuntu & Windows,
> consult [the documentation](https://onnxruntime.ai/docs/build/inferencing.html#prerequisites)
> for more information.

## Docs

For more information on this library, consult
the [documentation website](https://docs.ritual.net/ml-workflows/overview).

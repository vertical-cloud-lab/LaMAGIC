"""End-to-end tests that exercise the real LaMAGIC2 pipeline.

Unlike pure unit tests, these tests run the *actual* repository code against
the *real* released artifacts:

* the real LaMAGIC2 ``SFCI_345comp`` dataset from the Hugging Face Hub
  (https://huggingface.co/datasets/turtleben/LaMAGIC-dataset), and
* the real ``google/flan-t5-base`` tokenizer/config used by the training
  scripts.

They validate two core pieces of the repository:

1. The SFCI circuit *formulation* decoder in ``parsers/simulation.py`` — i.e.
   that a real model-style output string is parsed back into the correct
   circuit netlist and graph (the central contribution of the LaMAGIC2 paper).
2. The custom encoder-decoder transformer in
   ``analog_LLM/models/T5_transformer.py`` — i.e. that a real training
   forward/backward step (with the float ``vout``/``eff``/duty-cycle prefixes)
   produces a finite loss and gradients.

Network access is required. If the Hub cannot be reached the tests skip with a
clear message rather than failing.

Run with::

    pip install -r requirements-test.txt
    pytest tests/test_lamagic_pipeline.py
"""

import contextlib
import io
import json
import os

import pytest

DATASET_REPO = "turtleben/LaMAGIC-dataset"
DATASET_FILE = "transformed/LaMAGIC2/SFCI_345comp.json"
BASE_MODEL = "google/flan-t5-base"

# Keep the test fast: only a slice of the (132k-entry) dataset is inspected.
NUM_PARSE_SAMPLES = 200

DUTY_CYCLE_MAP = {
    "<duty_0.1>": 0.1,
    "<duty_0.3>": 0.3,
    "<duty_0.5>": 0.5,
    "<duty_0.7>": 0.7,
    "<duty_0.9>": 0.9,
}


def _suppress_stdout():
    """The parser/model code is very chatty; silence it during tests."""
    return contextlib.redirect_stdout(io.StringIO())


def _input_components(input_field):
    """Extract the set of device names (e.g. ``Sa0``) declared in an SFCI input.

    The SFCI ``input`` lists the circuit's ports and devices, e.g.
    ``"VIN VOUT GND Sa 0  Sa 1  Sb 2  L 3  L 4  <sep>"``. Device names are
    split across two whitespace-separated tokens (type + index).
    """
    tokens = input_field.replace("<sep>", "").split()
    components = set()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("VIN", "VOUT", "GND"):
            i += 1
        else:
            components.add(token + tokens[i + 1])
            i += 2
    return components


@pytest.fixture(scope="session")
def sfci_dataset():
    """Download the real LaMAGIC2 SFCI dataset and return a slice of entries."""
    hf_hub_download = pytest.importorskip("huggingface_hub").hf_hub_download
    try:
        path = hf_hub_download(
            DATASET_REPO,
            DATASET_FILE,
            repo_type="dataset",
            local_dir=os.environ.get("LAMAGIC_TEST_CACHE", "/tmp/lamagic_test_data"),
        )
    except Exception as exc:  # network / hub unavailable
        pytest.skip(f"Could not download {DATASET_REPO}/{DATASET_FILE}: {exc}")

    with open(path, "r") as handle:
        data = json.load(handle)

    assert len(data) > NUM_PARSE_SAMPLES
    return data[:NUM_PARSE_SAMPLES]


def test_dataset_schema(sfci_dataset):
    """The real dataset entries expose the expected conditional-generation keys."""
    sample = sfci_dataset[0]
    assert set(sample.keys()) >= {"vout", "eff", "input", "output"}
    assert isinstance(sample["vout"], float)
    assert isinstance(sample["eff"], float)
    assert sample["output"].split()[0] in DUTY_CYCLE_MAP


def test_sfci_parser_reconstructs_components(sfci_dataset):
    """The real SFCI decoder rebuilds exactly the devices declared in the input."""
    from parsers.simulation import read_transformer_output_shrink_canonical

    checked = 0
    for entry in sfci_dataset:
        out_stream = "<pad> " + entry["output"]
        with _suppress_stdout():
            netlist, duty = read_transformer_output_shrink_canonical(
                out_stream, duty10=False, typeNidx=True
            )

        # Every device in the input must appear in the parsed netlist, and no
        # spurious devices may be invented.
        assert set(netlist.keys()) == _input_components(entry["input"])

        # The decoded duty cycle must match the leading <duty_*> token.
        assert duty == DUTY_CYCLE_MAP[entry["output"].split()[0]]
        checked += 1

    assert checked == len(sfci_dataset)


def test_sfci_netlist_builds_connected_graph(sfci_dataset):
    """Parsed netlists form a single connected circuit with all three terminals."""
    nx = pytest.importorskip("networkx")
    from parsers.simulation import (
        convert_netlist_2_graph,
        read_transformer_output_shrink_canonical,
    )

    node_tokens = set()
    for device in ("Sa", "Sb", "C", "L"):
        for i in range(8):
            node_tokens.add(device + str(i))
    node_tokens.update(["IN", "OUT", "0"])

    for entry in sfci_dataset:
        out_stream = "<pad> " + entry["output"]
        with _suppress_stdout():
            netlist, _ = read_transformer_output_shrink_canonical(
                out_stream, duty10=False, typeNidx=True
            )
            graph = convert_netlist_2_graph(node_tokens, netlist)

        # IN/OUT/GND must all be present and the topology must be connected.
        assert {"IN", "OUT", "0"} <= set(graph.nodes)
        assert nx.is_connected(graph)


def test_model_training_step_on_real_data(sfci_dataset):
    """A real forward/backward step of the custom T5 yields a finite loss + grads."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    try:
        config = transformers.T5Config.from_pretrained(BASE_MODEL)
        tokenizer = transformers.T5Tokenizer.from_pretrained(
            BASE_MODEL,
            model_max_length=512,
            padding_side="right",
            use_fast=False,
            legacy=True,
        )
    except Exception as exc:  # network / hub unavailable
        pytest.skip(f"Could not download {BASE_MODEL}: {exc}")

    from analog_LLM.models.T5_transformer import (
        T5ForConditionalGeneration as T5Transformer,
    )

    # Shrink the model so the step runs quickly on CPU. This mirrors the
    # "train from scratch" path of the SFCI experiment, which overrides
    # ``d_model``/``vocab_size`` on the flan-t5-base config.
    config.d_model = 32
    config.d_kv = 8
    config.num_heads = 4
    config.d_ff = 64
    config.num_layers = 2
    config.num_decoder_layers = 2
    config.dropout_rate = 0.1
    config.vocab_size = len(tokenizer)

    with _suppress_stdout():
        model = T5Transformer(config=config)
        model.resize_token_embeddings(len(tokenizer))
    model.train()

    entry = sfci_dataset[0]
    input_ids = torch.tensor([tokenizer.encode(" ".join(entry["input"].split()))])
    labels = torch.tensor([tokenizer.encode(" ".join(entry["output"].split()))])
    vout = torch.tensor([[entry["vout"]]], dtype=torch.float)
    eff = torch.tensor([[entry["eff"]]], dtype=torch.float)
    # SFCI conditions generation on the five candidate duty-cycle options.
    d_cycle_option = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]], dtype=torch.float)

    with _suppress_stdout():
        output = model(
            input_ids=input_ids,
            labels=labels,
            vout=vout,
            eff=eff,
            d_cycle_option=d_cycle_option,
        )

    loss = output.loss
    assert torch.isfinite(loss)
    assert output.logits.shape[0] == 1
    assert output.logits.shape[-1] == config.vocab_size

    loss.backward()
    grad_total = sum(
        p.grad.abs().sum() for p in model.parameters() if p.grad is not None
    )
    assert torch.isfinite(grad_total)
    assert grad_total > 0

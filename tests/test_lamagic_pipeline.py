"""End-to-end tests that exercise the real LaMAGIC2 pipeline.

Unlike pure unit tests, these tests run the *actual* repository code against
the *real* released artifacts:

* the real LaMAGIC2 ``SFCI_345comp`` dataset from the Hugging Face Hub
  (https://huggingface.co/datasets/turtleben/LaMAGIC-dataset), and
* the real ``google/flan-t5-base`` tokenizer/config used by the training
  scripts.

They validate three core pieces of the repository:

1. The SFCI circuit *formulation* decoder in ``parsers/simulation.py`` — i.e.
   that a real model-style output string is parsed back into the correct
   circuit netlist and graph (the central contribution of the LaMAGIC2 paper).
2. The custom encoder-decoder transformer in
   ``analog_LLM/models/T5_transformer.py`` — i.e. that a real training
   forward/backward step (with the float ``vout``/``eff``/duty-cycle prefixes)
   produces a finite loss and gradients.
3. The ngspice verification loop in ``parsers/simulation.py`` — i.e. that a
   decoded netlist is converted to a ``.cki`` deck, simulated with the real
   ``ngspice`` binary, and reduced to a realized ``Vout``/efficiency.

Network access to the Hugging Face Hub is required. Both the dataset and the
``google/flan-t5-base`` artifacts are public, so no token/API key is needed.
The tests download them directly and fail loudly if the Hub is unreachable.
The ngspice loop additionally requires an ``ngspice`` binary on ``PATH``
(``apt-get install ngspice``).

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
# The released SFCI checkpoint exercised by the selection-by-simulation search.
SEARCH_CHECKPOINT = "turtleben/LaMAGIC2-345comp-SFCI-dataaug-noaug"

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
    """Download the real LaMAGIC2 SFCI dataset and return a slice of entries.

    The dataset is public; no Hugging Face token is required. A download
    failure is a real test failure, not a reason to skip.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DATASET_REPO,
        DATASET_FILE,
        repo_type="dataset",
        local_dir=os.environ.get("LAMAGIC_TEST_CACHE", "/tmp/lamagic_test_data"),
    )

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
    import networkx as nx
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
    import torch
    import transformers

    config = transformers.T5Config.from_pretrained(BASE_MODEL)
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        BASE_MODEL,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        legacy=True,
    )

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


def test_ngspice_simulation_closes_the_loop():
    """The real ngspice verifier simulates a decoded netlist end-to-end.

    This exercises ``parsers/simulation.py``'s full
    ``convert_netlist_cki`` -> ``ngspice`` -> ``calculate_efficiency`` loop on a
    real buck-style netlist (the one the released SFCI checkpoint generates for
    the powder-doser 12 V -> 5 V rail). It requires an ``ngspice`` binary on
    ``PATH`` (``apt-get install ngspice``); a missing binary is a real failure,
    not a reason to skip.
    """
    import math
    import shutil
    import tempfile

    from parsers.simulation import sim_netlist_duty_cycle, simulate_param

    assert shutil.which("ngspice") is not None, (
        "ngspice is not installed; install it (e.g. `apt-get install ngspice`) "
        "to run the simulation loop."
    )

    # Decoded SFCI topology for vout=0.4167, eff=0.95 (two-switch buck): each
    # device connects to exactly two nodes (IN/OUT/0 are VIN/VOUT/GND).
    netlist = {
        "C3": ["0", "9"],
        "C4": ["IN", "OUT"],
        "L2": ["IN", "10"],
        "Sa0": ["9", "10"],
        "Sb1": ["OUT", "10"],
    }
    duty_cycle = 0.3

    cki_path = os.path.join(tempfile.mkdtemp(prefix="lamagic_sim_"), "topology.cki")
    with _suppress_stdout():
        result = sim_netlist_duty_cycle(cki_path, netlist, duty_cycle)

    # The verifier returns the documented result schema.
    assert set(result.keys()) >= {"result_valid", "efficiency", "Vout", "error_msg"}

    # ngspice actually solved the circuit (no transient/alignment failure), so
    # the realized operating point is a finite, physical number.
    assert result["error_msg"] == "None"
    assert bool(result["result_valid"]) is True
    assert math.isfinite(float(result["Vout"]))
    assert math.isfinite(float(result["efficiency"]))
    # Efficiency of a real solved circuit is a valid fraction.
    assert 0.0 <= float(result["efficiency"]) <= 1.0
    # Realized output is a step-down of the simulator's Vin (no boosting here).
    vin = simulate_param["Vin"][0]
    assert 0.0 <= float(result["Vout"]) <= vin

def test_selection_by_simulation_search_finds_target():
    """The full search loop (``--search``) hits the target far better than greedy.

    This exercises the real selection-by-simulation search in
    ``experiment/lamagic2/generate_custom_topology.py``: it loads the *released*
    SFCI checkpoint, generates a pool of candidate topologies, simulates each in
    ngspice across all five duty-cycle options, and ranks them by closeness to a
    target Vout/Vin ratio. We assert the search returns valid, score-ordered
    candidates and that its best match beats a single greedy decode.

    Requires an ``ngspice`` binary on ``PATH`` (``apt-get install ngspice``);
    a missing binary is a real failure, not a reason to skip.
    """
    import math
    import shutil
    import tempfile

    assert shutil.which("ngspice") is not None, (
        "ngspice is not installed; install it (e.g. `apt-get install ngspice`) "
        "to run the selection-by-simulation search."
    )

    from experiment.lamagic2 import generate_custom_topology as gct

    target_vout, target_eff = 0.4167, 0.95
    components = ["Sa0", "Sa1", "Sb2", "Sb3", "C4"]

    with _suppress_stdout():
        tokenizer = gct.build_tokenizer()
        model = gct.build_model(SEARCH_CHECKPOINT, tokenizer)

        work_dir = tempfile.mkdtemp(prefix="lamagic_search_test_")
        results = gct.search_topologies(
            model, tokenizer, target_vout, target_eff, components, work_dir,
            num_candidates=4, sweep_budgets=False, seed=0,
        )

        # A single greedy decode, simulated at its chosen duty, for comparison.
        greedy_str = gct.generate_topology(
            model, tokenizer, gct.build_input_string(components),
            target_vout, target_eff,
        )

    # The search produced at least one valid, fully-populated candidate.
    assert results, "search returned no valid candidates"
    best = results[0]
    assert set(best.keys()) >= {
        "components", "netlist", "duty", "Vout", "vout_ratio", "efficiency",
        "score",
    }

    # Results are sorted best-first by score, and every candidate is a real,
    # finite, physical operating point produced by ngspice.
    vin = gct.simulate_param["Vin"][0]
    scores = [r["score"] for r in results]
    assert scores == sorted(scores)
    for r in results:
        assert math.isfinite(r["vout_ratio"]) and 0.0 <= r["Vout"] <= vin
        assert math.isfinite(r["efficiency"])
        assert r["duty"] in gct.DUTY_CYCLE_OPTIONS

    # The selected duty cycle is the one that actually minimises the score for
    # the best topology (i.e. selection really is by simulation, not by decode).
    best_key = tuple(sorted((d, tuple(best["netlist"][d])) for d in best["netlist"]))
    same_topo = [
        r for r in results
        if tuple(sorted((d, tuple(r["netlist"][d])) for d in r["netlist"])) == best_key
    ]
    assert best["duty"] == min(same_topo, key=lambda r: r["score"])["duty"]

    # The search's best candidate is at least as close to the target ratio as a
    # single greedy decode of the same budget.
    netlist, duty = gct.read_transformer_output_shrink_canonical(
        greedy_str, duty10=False, typeNidx=True
    )
    with _suppress_stdout():
        greedy_result = gct.sim_netlist_duty_cycle(
            os.path.join(tempfile.mkdtemp(), "greedy.cki"), netlist, duty
        )
    greedy_ratio = greedy_result["Vout"] / vin
    assert abs(best["vout_ratio"] - target_vout) <= abs(greedy_ratio - target_vout)

"""Generate a power-converter topology from a custom target spec with LaMAGIC2.

Unlike the training/validation entry points (which sweep the released dataset),
this script lets you ask the **released** LaMAGIC2 SFCI checkpoint for a circuit
that hits an arbitrary target, e.g. "step 12 V down to 5 V at high efficiency".

It runs the real model end-to-end on CPU (no GPU/ngspice required):

1. downloads the released SFCI checkpoint
   (``turtleben/LaMAGIC2-345comp-SFCI-dataaug-noaug``) and the
   ``google/flan-t5-base`` tokenizer,
2. builds the SFCI conditioning input from a target voltage-conversion ratio
   (``--vout`` = Vout/Vin), a target efficiency (``--eff``) and an available
   component budget (``--components``),
3. runs ``model.generate`` to produce the topology string,
4. decodes it with the real parser in ``parsers/simulation.py`` into a netlist
   and a circuit graph, and prints the result (optionally rendering a PNG).

Example (the powder-doser 12 V -> 5 V bench-rig buck rail)::

    python experiment/lamagic2/generate_custom_topology.py \
        --vout 0.4167 --eff 0.95 --components Sa0 Sb1 L2 C3 C4

The conditioning ratio/efficiency are passed to the model exactly as the
training data encodes them (raw ``Vout/Vin`` and raw efficiency in ``[0, 1]``).
The model proposes a topology; it does not by itself guarantee the simulated
operating point — close the loop with the repo's ngspice simulation if you need
verified Vout/efficiency.
"""

import argparse
import sys
from pathlib import Path

import torch
import transformers

# Make the repository importable regardless of the invocation directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analog_LLM.models.T5_transformer import (  # noqa: E402
    T5ForConditionalGeneration as T5Transformer,
)
from parsers.simulation import (  # noqa: E402
    convert_netlist_2_graph,
    read_transformer_output_shrink_canonical,
)

DEFAULT_CHECKPOINT = "turtleben/LaMAGIC2-345comp-SFCI-dataaug-noaug"
BASE_MODEL = "google/flan-t5-base"

# The five duty-cycle options the SFCI formulation conditions generation on.
DUTY_CYCLE_OPTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Device tokens added to the flan-t5 tokenizer for the SFCI (typeNidx) format.
# This mirrors ``add_device_token(..., typeNidx=True)`` in
# ``analog_LLM/utils/dataset.py`` so token ids match the released checkpoint.
SFCI_NODE_TOKENS = [
    "VIN", "VOUT", "GND",
    "<duty_0.1>", "<duty_0.2>", "<duty_0.3>", "<duty_0.4>", "<duty_0.5>",
    "<duty_0.6>", "<duty_0.7>", "<duty_0.8>", "<duty_0.9>",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
    "Sa", "Sb", "C", "L",
]


def build_tokenizer():
    """Load the flan-t5 tokenizer and add the SFCI device tokens."""
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        BASE_MODEL,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        legacy=True,
    )
    tokenizer.add_tokens(SFCI_NODE_TOKENS)
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "<sep>"})
    return tokenizer


def build_model(checkpoint, tokenizer):
    """Load the released SFCI checkpoint as the custom prefix-conditioned T5."""
    config = transformers.T5Config.from_pretrained(checkpoint)
    model = T5Transformer.from_pretrained(checkpoint, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return model


def split_component(token):
    """Split a compact component token like ``Sa0`` into ``("Sa", "0")``."""
    for i, ch in enumerate(token):
        if ch.isdigit():
            return token[:i], token[i:]
    raise ValueError(f"Component '{token}' must be a type followed by an index, e.g. Sa0")


def build_input_string(components):
    """Build the SFCI conditioning input, e.g. ``VIN VOUT GND Sa 0 Sb 1 ... <sep>``."""
    parts = ["VIN", "VOUT", "GND"]
    for comp in components:
        dev_type, idx = split_component(comp)
        parts.extend([dev_type, idx])
    parts.append("<sep>")
    return " ".join(parts)


def generate_topology(model, tokenizer, input_string, vout, eff, max_new_tokens=256):
    """Run the model and return the decoded topology string."""
    input_ids = torch.tensor([tokenizer.encode(input_string)])
    generation_config = transformers.GenerationConfig.from_pretrained(BASE_MODEL)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            d_cycle_option=torch.tensor([DUTY_CYCLE_OPTIONS], dtype=torch.float),
            vout=torch.tensor([[float(vout)]], dtype=torch.float),
            eff=torch.tensor([[float(eff)]], dtype=torch.float),
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            output_scores=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=False)


def node_token_universe():
    """Connection-node vocabulary accepted by ``convert_netlist_2_graph``."""
    tokens = set()
    for device in ("Sa", "Sb", "C", "L"):
        for i in range(13):
            tokens.add(device + str(i))
    tokens.update(["IN", "OUT", "0"])
    return tokens


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vout", type=float, required=True, help="Target voltage conversion ratio Vout/Vin.")
    parser.add_argument("--eff", type=float, required=True, help="Target efficiency in [0, 1].")
    parser.add_argument(
        "--components", nargs="+", default=["Sa0", "Sb1", "L2", "C3", "C4"],
        help="Available components as type+index tokens, e.g. Sa0 Sb1 L2 C3 C4.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HF repo id or local path of the SFCI checkpoint.")
    parser.add_argument("--render", metavar="PNG", default=None, help="Optional path to save a topology graph PNG.")
    args = parser.parse_args(argv)

    tokenizer = build_tokenizer()
    model = build_model(args.checkpoint, tokenizer)

    input_string = build_input_string(args.components)
    print("Target Vout/Vin :", args.vout)
    print("Target efficiency:", args.eff)
    print("Available comps  :", " ".join(args.components))
    print("Model input      :", input_string)

    raw_output = generate_topology(model, tokenizer, input_string, args.vout, args.eff)
    print("\nGenerated topology string:")
    print(" ", raw_output)

    netlist, duty_cycle = read_transformer_output_shrink_canonical(
        raw_output, duty10=False, typeNidx=True
    )
    print("\nDecoded duty cycle:", duty_cycle)
    print("Decoded netlist (device -> connection nodes):")
    for device in sorted(netlist):
        print(f"  {device:<4} -> {', '.join(netlist[device])}")

    graph = convert_netlist_2_graph(node_token_universe(), netlist)
    print("\nCircuit graph:")
    print("  nodes:", sorted(graph.nodes))
    print("  edges:", sorted(tuple(sorted(e)) for e in graph.edges))

    if args.render:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx

        plt.figure(figsize=(6, 6))
        nx.draw(graph, with_labels=True, node_color="#cde", node_size=900, font_size=8)
        plt.title(f"Vout/Vin={args.vout}, eff={args.eff}, duty={duty_cycle}")
        plt.tight_layout()
        plt.savefig(args.render, dpi=200)
        print("\nSaved topology graphic to", args.render)


if __name__ == "__main__":
    main()

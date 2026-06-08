"""Generate a power-converter topology from a custom target spec with LaMAGIC2.

Unlike the training/validation entry points (which sweep the released dataset),
this script lets you ask the **released** LaMAGIC2 SFCI checkpoint for a circuit
that hits an arbitrary target, e.g. "step 12 V down to 5 V at high efficiency".

It runs the real model end-to-end on CPU (no GPU required):

1. downloads the released SFCI checkpoint
   (``turtleben/LaMAGIC2-345comp-SFCI-dataaug-noaug``) and the
   ``google/flan-t5-base`` tokenizer,
2. builds the SFCI conditioning input from a target voltage-conversion ratio
   (``--vout`` = Vout/Vin), a target efficiency (``--eff``) and an available
   component budget (``--components``),
3. runs ``model.generate`` to produce the topology string,
4. decodes it with the real parser in ``parsers/simulation.py`` into a netlist
   and a circuit graph, and prints the result (optionally rendering a PNG), and
5. optionally (``--simulate``) closes the loop with the repository's ngspice
   simulation to report the *actually realized* ``Vout/Vin`` ratio and
   efficiency of the generated topology.

Example (single greedy decode of the powder-doser 12 V -> 5 V buck rail)::

    python experiment/lamagic2/generate_custom_topology.py \
        --vout 0.4167 --eff 0.95 --components Sa0 Sb1 L2 C3 C4 --simulate

The conditioning ratio/efficiency are passed to the model exactly as the
training data encodes them (raw ``Vout/Vin`` and raw efficiency in ``[0, 1]``).
A single greedy decode proposes one topology; it does not by itself guarantee
the simulated operating point. Pass ``--simulate`` to verify the realized
Vout/efficiency with ngspice (requires an ``ngspice`` install, e.g.
``apt-get install ngspice``).

Full selection-by-simulation search (``--search``)
--------------------------------------------------
This mirrors how LaMAGIC2 is meant to be used in practice: instead of trusting
a single decode, it *generates many candidate topologies, simulates each one in
ngspice across the five conditioning duty-cycle options, and selects the
candidate + duty cycle whose realized Vout/efficiency are closest to the
target*. It sweeps three axes:

* **Topology candidates** -- the model is sampled ``--num-candidates`` times
  (plus one greedy decode) to produce a diverse pool of topologies.
* **Component budgets** -- with ``--sweep-budgets`` it additionally sweeps the
  most common 3/4/5-component budgets from the released dataset, not just the
  one passed via ``--components``.
* **Duty cycle** -- every candidate topology is simulated at all five duty-cycle
  options (0.1/0.3/0.5/0.7/0.9), since the operating point depends strongly on
  the duty cycle.

Example (full search for the powder-doser 12 V -> 5 V buck rail)::

    python experiment/lamagic2/generate_custom_topology.py \
        --vout 0.4167 --eff 0.95 --components Sa0 Sb1 L2 C3 C4 \
        --search --sweep-budgets

``--search`` requires an ``ngspice`` install (e.g. ``apt-get install ngspice``).
"""

import argparse
import shutil
import sys
import tempfile
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
    simulate_param,
    sim_netlist_duty_cycle,
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
    """Run the model and return the decoded topology string (greedy)."""
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


def generate_candidates(model, tokenizer, input_string, vout, eff,
                        num_candidates=8, top_k=20, temperature=1.0,
                        max_new_tokens=256):
    """Return a de-duplicated pool of candidate topology strings.

    The pool is one greedy decode plus ``num_candidates`` sampled decodes, which
    gives the selection-by-simulation search a diverse set of topologies to try.
    """
    strings = [generate_topology(model, tokenizer, input_string, vout, eff,
                                 max_new_tokens=max_new_tokens)]
    if num_candidates > 0:
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
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=num_candidates,
            )
        for row in output:
            strings.append(tokenizer.decode(row, skip_special_tokens=False))
    # De-duplicate while preserving order.
    seen, unique = set(), []
    for s in strings:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


# The most common 3/4/5-component budgets in the released SFCI_345comp dataset,
# expressed as device-type counts. Indices are assigned globally in
# (Sa, Sb, L, C) order to match the dataset's ``Sa 0 Sb 1 L 2 ...`` encoding.
DEFAULT_SWEEP_BUDGETS = [
    {"Sa": 1, "Sb": 2, "L": 1, "C": 1},
    {"Sa": 2, "Sb": 1, "L": 1, "C": 1},
    {"Sa": 1, "Sb": 1, "L": 2, "C": 1},
    {"Sa": 1, "Sb": 1, "L": 1, "C": 2},
    {"Sa": 2, "Sb": 2, "L": 0, "C": 1},
    {"Sa": 2, "Sb": 2, "L": 1, "C": 0},
]


def budget_to_components(counts):
    """Turn a ``{type: n}`` budget into ordered tokens like ``["Sa0", "Sb1", ...]``.

    Indices are assigned globally in (Sa, Sb, L, C) order to match the released
    dataset encoding (e.g. ``Sa 0 Sb 1 L 2 C 3 C 4``).
    """
    components, idx = [], 0
    for dev_type in ("Sa", "Sb", "L", "C"):
        for _ in range(counts.get(dev_type, 0)):
            components.append(f"{dev_type}{idx}")
            idx += 1
    return components


def score_candidate(ratio, efficiency, target_vout, target_eff, eff_weight):
    """Distance of a simulated operating point from the target (lower is better).

    The voltage-conversion ratio is the primary objective; efficiency is added
    with ``eff_weight`` so it acts mainly as a tie-breaker between topologies
    that hit the ratio.
    """
    return abs(ratio - target_vout) + eff_weight * abs(efficiency - target_eff)


def search_topologies(model, tokenizer, target_vout, target_eff, components,
                      work_dir, num_candidates=8, duty_options=None,
                      sweep_budgets=False, eff_weight=0.25, seed=0,
                      max_new_tokens=256):
    """Selection-by-simulation search over topologies, budgets and duty cycles.

    Generates candidate topologies (sampled + greedy) for each component budget,
    simulates every candidate at each duty-cycle option with ngspice, and ranks
    the (topology, duty) pairs by how close the realized Vout/efficiency are to
    the target. Returns a list of result dicts sorted best-first.
    """
    if shutil.which("ngspice") is None:
        raise RuntimeError(
            "ngspice is not installed; install it to run the selection-by-"
            "simulation search (e.g. `apt-get install ngspice` or `conda "
            "install -c conda-forge ngspice`)."
        )
    if duty_options is None:
        duty_options = list(DUTY_CYCLE_OPTIONS)
    torch.manual_seed(seed)

    budgets = []
    base = " ".join(components)
    budgets.append(list(components))
    if sweep_budgets:
        for counts in DEFAULT_SWEEP_BUDGETS:
            comps = budget_to_components(counts)
            if " ".join(comps) != base:
                budgets.append(comps)

    node_tokens = node_token_universe()
    vin = simulate_param["Vin"][0]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    results, seen_netlists, sim_id = [], set(), 0
    for budget in budgets:
        input_string = build_input_string(budget)
        candidate_strings = generate_candidates(
            model, tokenizer, input_string, target_vout, target_eff,
            num_candidates=num_candidates, max_new_tokens=max_new_tokens,
        )
        for raw_output in candidate_strings:
            try:
                netlist, model_duty = read_transformer_output_shrink_canonical(
                    raw_output, duty10=False, typeNidx=True
                )
                graph = convert_netlist_2_graph(node_tokens, netlist)
            except Exception:
                continue
            # Skip topologies we've already simulated (across budgets/samples).
            key = tuple(sorted((d, tuple(netlist[d])) for d in netlist))
            if key in seen_netlists:
                continue
            seen_netlists.add(key)
            for duty in duty_options:
                cki_path = work_dir / f"search_{sim_id}.cki"
                sim_id += 1
                try:
                    result = sim_netlist_duty_cycle(str(cki_path), netlist, duty)
                except Exception:
                    continue
                if not result.get("result_valid"):
                    continue
                realized = result.get("Vout")
                eff = result.get("efficiency")
                if realized is None or eff is None:
                    continue
                ratio = realized / vin if vin else float("nan")
                results.append({
                    "components": list(budget),
                    "input": input_string,
                    "raw_output": raw_output,
                    "netlist": netlist,
                    "graph": graph,
                    "model_duty": model_duty,
                    "duty": duty,
                    "Vout": realized,
                    "vout_ratio": ratio,
                    "efficiency": eff,
                    "score": score_candidate(ratio, eff, target_vout,
                                             target_eff, eff_weight),
                })
    results.sort(key=lambda r: r["score"])
    return results


def node_token_universe():
    """Connection-node vocabulary accepted by ``convert_netlist_2_graph``."""
    tokens = set()
    for device in ("Sa", "Sb", "C", "L"):
        for i in range(13):
            tokens.add(device + str(i))
    tokens.update(["IN", "OUT", "0"])
    return tokens


def simulate_topology(netlist, duty_cycle, cki_path):
    """Close the loop with ngspice: simulate the netlist and report the result.

    Returns the result dict from ``parsers.simulation.calculate_efficiency``,
    augmented with the realized ``Vout/Vin`` ratio. Requires an ``ngspice``
    binary on ``PATH`` (e.g. ``apt-get install ngspice``).
    """
    if shutil.which("ngspice") is None:
        raise RuntimeError(
            "ngspice is not installed; install it to simulate generated "
            "topologies (e.g. `apt-get install ngspice` or `conda install -c "
            "conda-forge ngspice`)."
        )
    result = sim_netlist_duty_cycle(str(cki_path), netlist, duty_cycle)
    vin = simulate_param["Vin"][0]
    realized = result.get("Vout", float("nan"))
    result["vout_ratio"] = realized / vin if vin else float("nan")
    return result


def run_search(args, model, tokenizer):
    """Run --search: rank candidates by simulated closeness, print and render best."""
    work_dir = args.cki_path
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="lamagic_search_"))
    else:
        work_dir = Path(work_dir)
        if work_dir.suffix:  # a file path was given; use its parent directory
            work_dir = work_dir.parent

    print("Target Vout/Vin :", args.vout)
    print("Target efficiency:", args.eff)
    print("Available comps  :", " ".join(args.components))
    print("Sweep budgets    :", args.sweep_budgets)
    print("Candidates/budget:", args.num_candidates, "(sampled) + 1 greedy")
    print("Duty options     :", DUTY_CYCLE_OPTIONS)
    print("\nRunning selection-by-simulation search (this may take a while)...")

    results = search_topologies(
        model, tokenizer, args.vout, args.eff, args.components, work_dir,
        num_candidates=args.num_candidates, sweep_budgets=args.sweep_budgets,
        eff_weight=args.eff_weight, seed=args.seed,
    )

    if not results:
        print("\nNo valid topology was found. Try increasing --num-candidates "
              "or enabling --sweep-budgets.")
        return

    vin = simulate_param["Vin"][0]
    n_show = min(args.top, len(results))
    print(f"\nSimulated {len(results)} valid (topology, duty) candidates. "
          f"Top {n_show} by closeness to target:")
    print("  %-4s %-7s %-10s %-10s %-8s %s"
          % ("rank", "duty", "Vout/Vin", "eff", "score", "components"))
    for rank, r in enumerate(results[:n_show], 1):
        print("  %-4d %-7.1f %-10.4f %-10.4f %-8.4f %s"
              % (rank, r["duty"], r["vout_ratio"], r["efficiency"], r["score"],
                 " ".join(r["components"])))

    best = results[0]
    print("\nBest topology (rank 1):")
    print("  components         :", " ".join(best["components"]))
    print("  selected duty cycle:", best["duty"], "(model proposed",
          best["model_duty"], "for the greedy decode of this budget)")
    print("  realized Vout      :", best["Vout"])
    print("  realized Vout/Vin  :", best["vout_ratio"], "(target", args.vout, ")")
    print("  realized efficiency:", best["efficiency"], "(target", args.eff, ")")
    print("  netlist (device -> connection nodes):")
    for device in sorted(best["netlist"]):
        print(f"    {device:<4} -> {', '.join(best['netlist'][device])}")
    graph = best["graph"]
    print("  circuit graph nodes:", sorted(graph.nodes))
    print("  circuit graph edges:", sorted(tuple(sorted(e)) for e in graph.edges))

    if args.render:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx

        plt.figure(figsize=(6, 6))
        nx.draw(graph, with_labels=True, node_color="#cde", node_size=900, font_size=8)
        plt.title("Vout/Vin=%.4f (target %.4f), eff=%.3f, duty=%g"
                  % (best["vout_ratio"], args.vout, best["efficiency"], best["duty"]))
        plt.tight_layout()
        plt.savefig(args.render, dpi=200)
        print("\nSaved best-topology graphic to", args.render)


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
    parser.add_argument(
        "--simulate", action="store_true",
        help="Close the loop with ngspice to report the realized Vout/efficiency "
             "(requires an ngspice install).",
    )
    parser.add_argument(
        "--cki-path", default=None,
        help="Where to write the ngspice .cki netlist used by --simulate "
             "(default: a temporary file).",
    )
    parser.add_argument(
        "--search", action="store_true",
        help="Run the full selection-by-simulation search: generate many "
             "candidate topologies, simulate each across all duty-cycle options "
             "with ngspice, and select the one closest to the target "
             "(requires an ngspice install).",
    )
    parser.add_argument(
        "--sweep-budgets", action="store_true",
        help="With --search, also sweep the most common dataset component "
             "budgets in addition to --components.",
    )
    parser.add_argument(
        "--num-candidates", type=int, default=8,
        help="With --search, number of sampled topologies per component budget "
             "(in addition to one greedy decode). Default: 8.",
    )
    parser.add_argument(
        "--eff-weight", type=float, default=0.25,
        help="With --search, weight of the efficiency term relative to the "
             "Vout-ratio term when ranking candidates. Default: 0.25.",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="With --search, how many of the best-ranked candidates to print. "
             "Default: 5.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="With --search, random seed for sampled generation. Default: 0.",
    )
    args = parser.parse_args(argv)

    tokenizer = build_tokenizer()
    model = build_model(args.checkpoint, tokenizer)

    if args.search:
        run_search(args, model, tokenizer)
        return

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

    if args.simulate:
        cki_path = args.cki_path
        if cki_path is None:
            cki_path = Path(tempfile.mkdtemp(prefix="lamagic_sim_")) / "topology.cki"
        result = simulate_topology(netlist, duty_cycle, cki_path)
        vin = simulate_param["Vin"][0]
        print("\nngspice verification (Vin = %g V, duty = %g):" % (vin, duty_cycle))
        print("  realized Vout      :", result.get("Vout"))
        print("  realized Vout/Vin  :", result.get("vout_ratio"))
        print("  target  Vout/Vin   :", args.vout)
        print("  realized efficiency:", result.get("efficiency"))
        print("  target  efficiency :", args.eff)
        print("  result_valid       :", result.get("result_valid"))
        if result.get("error_msg") not in (None, "None"):
            print("  error_msg          :", result.get("error_msg"))

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

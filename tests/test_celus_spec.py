"""Tests for the Celus-compatible structured design specification.

These exercise :mod:`experiment.lamagic2.generate_celus_spec`, which turns the
abstract LaMAGIC2 power-stage topology into a structured electronic design spec
(real parts, pin-level connectivity, net classes, design rules, functional
blocks, interface protocols, and global requirements) for the powder-doser
bench rig. They run offline with no GPU/network and assert each of the eight
required spec sections is present and internally consistent.

Run with::

    pytest tests/test_celus_spec.py
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_SCRIPT = REPO_ROOT / "experiment" / "lamagic2" / "generate_celus_spec.py"
SPEC_JSON = REPO_ROOT / "experiment" / "lamagic2" / "results" / "powder_doser_celus_spec.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_celus_spec", SPEC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gcs():
    return _load_module()


@pytest.fixture(scope="module")
def spec(gcs):
    return gcs.build_spec()


def test_spec_is_self_consistent(gcs, spec):
    # The generator's own validator must find no problems (pins reference real
    # nets, components belong to declared blocks, interface nets exist, etc.).
    assert gcs.validate_spec(spec) == []


def test_spec_has_all_eight_requirement_sections(spec):
    # Requirement #8 global requirements, #6 blocks, #7 interfaces,
    # #2/#3 nets, #1/#4 components, #5 design rules, plus the abstract->concrete
    # power-stage mapping.
    for key in (
        "global_requirements",
        "functional_blocks",
        "interfaces",
        "nets",
        "components",
        "design_rules",
        "power_stage_realization",
    ):
        assert key in spec, f"missing spec section: {key}"


def test_components_have_typing_mpn_and_parametric_data(spec):
    # Requirement #1: explicit component typing + parametric data with real
    # manufacturer part numbers, replacing the anonymous LaMAGIC2 node labels.
    # (Concrete BOM designators like C3 are real parts with an MPN; the abstract
    # Sa0/Sb1/L2/C3/C4 nodes only live in ``power_stage_realization``.)
    for comp in spec["components"]:
        assert comp["type"]
        assert comp["mpn"]
        assert comp["description"]
        assert comp["parameters"], f"{comp['ref']} has no parameters"


def test_every_component_pin_maps_to_a_declared_net(spec):
    # Requirement #2: pin-level connectivity.
    net_names = {n["name"] for n in spec["nets"]}
    for comp in spec["components"]:
        assert comp["pins"], f"{comp['ref']} has no pins"
        for pin, net in comp["pins"].items():
            if net is not None:
                assert net in net_names, f"{comp['ref']}.{pin} -> {net}"


def test_power_and_ground_nets_are_classified(spec):
    # Requirement #3: GND + power-delivery net classes with current ratings.
    by_name = {n["name"]: n for n in spec["nets"]}
    assert by_name["GND"]["class"] == "ground"
    for rail in ("+12V", "+5V", "+3V3"):
        assert by_name[rail]["class"] == "power"
        assert by_name[rail]["nominal_v"] > 0
    # The 12 V high-current rail must carry a current rating for trace widths.
    assert by_name["+12V"]["max_current_a"] >= 5.0


def test_components_have_footprints(spec):
    # Requirement #4: footprint / packaging constraints.
    for comp in spec["components"]:
        assert comp["footprint"], f"{comp['ref']} has no footprint"


def test_design_rules_isolate_high_current_from_logic(spec):
    # Requirement #5: isolation/clearance separating the motor/12 V domain from
    # the low-voltage logic domain, plus net-class trace widths.
    rules = spec["design_rules"]
    assert rules["net_classes"]
    assert rules["isolation"]
    joined = " ".join(r["rule"] for r in rules["isolation"]).lower()
    assert "high-current" in joined or "high current" in joined
    # The 12 V power class must be wider than the logic-signal class.
    power = rules["net_classes"]["power_high"]["min_trace_width_mm"]
    logic = rules["net_classes"]["logic_signal"]["min_trace_width_mm"]
    assert power > logic


def test_functional_blocks_partition_all_components(spec):
    # Requirement #6: every component belongs to exactly one functional block.
    comp_refs = {c["ref"] for c in spec["components"]}
    seen = []
    for block in spec["functional_blocks"]:
        seen.extend(block["components"])
    assert sorted(seen) == sorted(comp_refs)
    assert len(seen) == len(set(seen)), "a component appears in multiple blocks"
    block_names = {b["id"] for b in spec["functional_blocks"]}
    assert {"power_management", "control_mcu",
            "motor_driver_stage", "sensor_haptic_interface"} <= block_names


def test_interfaces_cover_expected_protocols(spec):
    # Requirement #7: I2C, UART/TTL-serial, PWM and DC power-rail interfaces.
    protocols = {i["protocol"] for i in spec["interfaces"]}
    assert {"I2C", "TTL-serial", "PWM", "DC"} <= protocols


def test_global_requirements_capture_system_parameters(spec):
    # Requirement #8: input voltage, logic levels, max continuous currents.
    g = spec["global_requirements"]
    assert g["input_voltage_v"] == 12.0
    assert 3.3 in g["logic_levels_v"] and 5.0 in g["logic_levels_v"]
    assert g["max_continuous_current_a"]["+5V_buck"] == 2.5


def test_abstract_lamagic2_nodes_map_to_concrete_parts(spec):
    # The whole point of Luke's request: the abstract Sa0/Sb1/L2/C3/C4 power
    # stage must be tied to a concrete realization (the D24V22F5 buck).
    realization = spec["power_stage_realization"]["12V_to_5V"]
    assert set(realization["lamagic2_abstract_nodes"]) == {"Sa0", "Sb1", "L2", "C3", "C4"}
    concrete = realization["concrete_realization"]
    assert "D24V22F5" in concrete["module"]
    # And the buck component itself records which LaMAGIC2 stage it realizes.
    u1 = next(c for c in spec["components"] if c["ref"] == "U1")
    assert u1["realizes_lamagic2"] == "12V_to_5V"


def test_committed_json_matches_generator_output(gcs, spec):
    # The checked-in artifact must be exactly what the generator produces, so it
    # can't silently drift from the source data.
    assert SPEC_JSON.exists(), f"missing committed spec: {SPEC_JSON}"
    on_disk = json.loads(SPEC_JSON.read_text())
    assert on_disk == spec

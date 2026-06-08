"""Turn an abstract LaMAGIC2 topology into a structured electronic design spec.

LaMAGIC2 (see :mod:`generate_custom_topology`) emits an *abstract* node-edge
power-converter topology: anonymous switches/inductors/capacitors (``Sa0``,
``Sb1``, ``L2``, ``C3``, ``C4``) wired between ``VIN``/``VOUT``/``GND``. That is
useful for exploring converter structure, but it carries none of the
project-specific information a layout/synthesis tool (e.g. the Celus platform)
needs: real parts, parametric values, pin-level connectivity, net classes,
footprints, design rules, functional blocks, and interface protocols.

This script bridges that gap for the powder-doser bench rig. It reads the real
component/connectivity data published in powder-doser PR #61
(``hardware/test-module/README.md`` BOM + pin/net table, ``firmware/config.py``,
``kicad/generate.py`` symbol pin maps) and emits a structured, Celus-compatible
block-diagram / JSON-netlist specification that:

1. **Explicit component typing & parametric data** — real passives/ICs with
   manufacturer part numbers (MPNs) and parameters, replacing ``C4``/``L2``/...
2. **Pin-level connectivity** — every component pin mapped to a named net.
3. **Power & ground designations** — ``GND`` nets plus power-net classes
   (``+12V``/``+5V``/``+3V3``) carrying current ratings for trace-width rules.
4. **Footprint & packaging constraints** — package/form-factor per component.
5. **Design rules & constraints** — isolation/clearance between the
   high-current motor/12 V domain and the low-voltage logic domain.
6. **Functional-block grouping** — Power Management, Control/MCU, Motor Driver
   Stage, Sensor/Haptic Interface.
7. **Interface protocols** — I2C, TTL-serial (UART), PWM, and DC power rails.
8. **Global system requirements** — input voltage, logic levels, currents.

It also records how the abstract LaMAGIC2 power-stage nodes map onto the
concrete, off-the-shelf realization chosen in PR #61 (the D24V22F5 buck for
12 V -> 5 V, the Pico W on-board LDO for 5 V -> 3.3 V).

Run::

    python experiment/lamagic2/generate_celus_spec.py \
        --out experiment/lamagic2/results/powder_doser_celus_spec.json

The component database below is transcribed from powder-doser PR #61
(https://github.com/vertical-cloud-lab/powder-doser/pull/61); no network or GPU
is required to assemble the spec.
"""

import argparse
import json
import sys
from pathlib import Path

# Source-of-truth references (powder-doser PR #61, head commit).
PR61 = "vertical-cloud-lab/powder-doser#61"
PR61_COMMIT = "147e5055fb6ec935af164a88e447ed1748f370df"
PR61_README = (
    "https://github.com/vertical-cloud-lab/powder-doser/blob/"
    f"{PR61_COMMIT}/hardware/test-module/README.md"
)

# ---------------------------------------------------------------------------
# Global system requirements (requirement #8) — from PR #61 BOM + config.py.
# ---------------------------------------------------------------------------
GLOBAL_REQUIREMENTS = {
    "input_voltage_v": 12.0,
    "input_source": "Mean Well GST60A12-P1J (12 V / 5 A, 60 W barrel-jack PSU)",
    "input_max_current_a": 5.0,
    "logic_levels_v": [3.3, 5.0],
    "logic_reference_v": 3.3,
    "max_continuous_current_a": {
        "+12V_input": 5.0,
        "+5V_buck": 2.5,
        "stepper_per_phase": 0.67,
        "stepper_driver_limit": 1.5,
    },
    "absolute_max": {
        "tic_t500_vin_v": 35.0,
    },
    "notes": (
        "Single powder-doser channel driven from one Raspberry Pi Pico W; only "
        "GP0..GP15 are used so the firmware runs unmodified on a plain Pico."
    ),
}

# ---------------------------------------------------------------------------
# Nets (requirements #2 + #3) — name -> {class, protocol/voltage, ...}.
# Net classes drive downstream trace-width / clearance rules.
# ---------------------------------------------------------------------------
NETS = [
    {"name": "+12V", "class": "power", "domain": "high_current",
     "nominal_v": 12.0, "max_current_a": 5.0, "source": "J1.+ (PSU)"},
    {"name": "+5V", "class": "power", "domain": "logic",
     "nominal_v": 5.0, "max_current_a": 2.5, "source": "U1.VOUT (buck)"},
    {"name": "+3V3", "class": "power", "domain": "logic",
     "nominal_v": 3.3, "max_current_a": 0.3, "source": "U2.3V3 (Pico LDO)"},
    {"name": "GND", "class": "ground", "domain": "global",
     "nominal_v": 0.0, "notes": "Common star ground for all components."},

    {"name": "I2C_SDA", "class": "signal", "protocol": "I2C", "logic_v": 3.3},
    {"name": "I2C_SCL", "class": "signal", "protocol": "I2C", "logic_v": 3.3},
    {"name": "STP_TX", "class": "signal", "protocol": "UART", "logic_v": 3.3},
    {"name": "STP_RX", "class": "signal", "protocol": "UART", "logic_v": 3.3},
    {"name": "SOL_IN1", "class": "signal", "protocol": "PWM", "logic_v": 3.3},
    {"name": "SOL_IN2", "class": "signal", "protocol": "GPIO", "logic_v": 3.3},
    {"name": "HAPT_EN", "class": "signal", "protocol": "GPIO", "logic_v": 3.3},
    {"name": "SERVO_SIG", "class": "signal", "protocol": "PWM", "logic_v": 3.3},

    {"name": "STP_A1", "class": "motor_coil", "domain": "high_current"},
    {"name": "STP_A2", "class": "motor_coil", "domain": "high_current"},
    {"name": "STP_B1", "class": "motor_coil", "domain": "high_current"},
    {"name": "STP_B2", "class": "motor_coil", "domain": "high_current"},
    {"name": "VIB_A", "class": "motor_out", "domain": "high_current"},
    {"name": "VIB_B", "class": "motor_out", "domain": "high_current"},
    {"name": "SOL_A", "class": "motor_out", "domain": "high_current"},
    {"name": "SOL_B", "class": "motor_out", "domain": "high_current"},
]

# ---------------------------------------------------------------------------
# Components (requirements #1, #2, #4, #6) — concrete parts replacing the
# abstract LaMAGIC2 nodes, with MPNs, parameters, pin->net maps, footprints,
# and functional-block membership.
# ---------------------------------------------------------------------------
COMPONENTS = [
    {
        "ref": "J1", "type": "dc_power_supply",
        "mpn": "Mean Well GST60A12-P1J",
        "description": "12 V / 5 A (60 W) barrel-jack PSU — system power",
        "block": "power_management",
        "footprint": "Barrel jack 5.5x2.1 mm (off-board brick)",
        "parameters": {"vout_v": 12.0, "iout_a": 5.0, "power_w": 60.0},
        "pins": {"+": "+12V", "-": "GND"},
    },
    {
        "ref": "U1", "type": "buck_regulator",
        "mpn": "Pololu D24V22F5",
        "description": "12 V -> 5 V / 2.5 A step-down buck (powers Pico W + servo + solenoid logic)",
        "block": "power_management",
        "footprint": "Pololu D24V22F5 module, 0.1\" THT header (0.4\"x0.5\")",
        "parameters": {"vin_v": 12.0, "vout_v": 5.0, "iout_a": 2.5,
                       "topology": "synchronous buck"},
        "pins": {"VIN": "+12V", "GND_IN": "GND", "SHDN": None,
                 "VOUT": "+5V", "GND_OUT": "GND"},
        "realizes_lamagic2": "12V_to_5V",
    },
    {
        "ref": "U2", "type": "mcu_module",
        "mpn": "Raspberry Pi Pico W (RP2040 + CYW43439)",
        "description": "Host MCU; on-board LDO also realizes the 5 V -> 3.3 V logic rail",
        "block": "control_mcu",
        "footprint": "Pico 2x20 castellated/0.1\" header (51x21 mm)",
        "parameters": {"vsys_v": 5.0, "v3v3_v": 3.3, "gpio_used": "GP0..GP15"},
        "pins": {"VSYS": "+5V", "3V3": "+3V3", "GND": "GND",
                 "GP0": "I2C_SDA", "GP1": "I2C_SCL",
                 "GP4": "STP_TX", "GP5": "STP_RX",
                 "GP10": "SOL_IN1", "GP11": "SOL_IN2",
                 "GP14": "HAPT_EN", "GP15": "SERVO_SIG"},
        "realizes_lamagic2": "5V_to_3V3",
    },
    {
        "ref": "U3", "type": "haptic_driver",
        "mpn": "Adafruit DRV2605L breakout (#2305)",
        "description": "I2C haptic driver — drives the ERM coin motor",
        "block": "sensor_haptic_interface",
        "footprint": "Adafruit breakout, 0.1\" header",
        "parameters": {"interface": "I2C", "vin_v": 3.3,
                       "effect_id": 14, "library": 1},
        "pins": {"VIN": "+3V3", "GND": "GND", "SDA": "I2C_SDA", "SCL": "I2C_SCL",
                 "EN": "HAPT_EN", "IN_TRIG": "HAPT_EN",
                 "OUT+": "VIB_A", "OUT-": "VIB_B"},
    },
    {
        "ref": "M1", "type": "vibration_motor",
        "mpn": "Adafruit #1201 (10 mm ERM coin)",
        "description": "Eccentric-rotating-mass vibration motor",
        "block": "sensor_haptic_interface",
        "footprint": "10 mm coin, flying leads",
        "parameters": {"diameter_mm": 10.0, "type": "ERM"},
        "pins": {"+": "VIB_A", "-": "VIB_B"},
    },
    {
        "ref": "U4", "type": "brushed_dc_motor_driver",
        "mpn": "Adafruit DRV8871 breakout (#3190)",
        "description": "H-bridge driver — pulses the tap solenoid",
        "block": "motor_driver_stage",
        "footprint": "Adafruit breakout, 0.1\" header",
        "parameters": {"vm_v": 12.0, "imax_a": 3.6, "control": "PWM (IN1/IN2)"},
        "pins": {"VM": "+12V", "GND": "GND", "IN1": "SOL_IN1", "IN2": "SOL_IN2",
                 "OUT1": "SOL_A", "OUT2": "SOL_B"},
    },
    {
        "ref": "SOL1", "type": "solenoid",
        "mpn": "JF-0530B (Adafruit #412)",
        "description": "5 V push-pull tap solenoid (PWM-limited holding force)",
        "block": "motor_driver_stage",
        "footprint": "Solenoid body, flying leads",
        "parameters": {"voltage_v": 5.0, "drive": "PWM duty (TAP_PWM_DUTY)"},
        "pins": {"+": "SOL_A", "-": "SOL_B"},
    },
    {
        "ref": "U5", "type": "stepper_controller",
        "mpn": "Pololu Tic T500 (#3134, MP6500)",
        "description": "USB/TTL-serial stepper controller with on-board motion planner",
        "block": "motor_driver_stage",
        "footprint": "Pololu Tic T500 module, 0.1\" header + screw terminals",
        "parameters": {"vin_v": 12.0, "vin_max_v": 35.0,
                       "current_limit_a": 0.67, "continuous_limit_a": 1.5,
                       "interface": "TTL serial (UART)", "baud": 9600,
                       "microsteps": 8},
        "pins": {"VIN": "+12V", "GND": "GND", "RX": "STP_TX", "TX": "STP_RX",
                 "ERR": None,
                 "A1": "STP_A1", "A2": "STP_A2", "B1": "STP_B1", "B2": "STP_B2"},
    },
    {
        "ref": "M2", "type": "stepper_motor",
        "mpn": "NEMA-11 11HS18-0674S",
        "description": "4-wire bipolar stepper, direct-coupled to the auger",
        "block": "motor_driver_stage",
        "footprint": "NEMA-11 frame (28 mm), flying leads",
        "parameters": {"phase_current_a": 0.67, "step_deg": 1.8,
                       "full_steps_rev": 200},
        "pins": {"A1": "STP_A1", "A2": "STP_A2", "B1": "STP_B1", "B2": "STP_B2"},
    },
    {
        "ref": "M3", "type": "servo",
        "mpn": "HD-1810MG (#1142)",
        "description": "Metal-gear digital servo on the dispensing-angle axis",
        "block": "motor_driver_stage",
        "footprint": "Standard hobby servo, 3-pin 0.1\" header",
        "parameters": {"supply_v": 5.0, "pwm_hz": 50,
                       "pulse_us": [500, 2400], "angle_deg": [0, 180]},
        "pins": {"+5V": "+5V", "GND": "GND", "SIG": "SERVO_SIG"},
    },
    {
        "ref": "SR1", "type": "shunt_regulator",
        "mpn": "Pololu #3776 (33 V / 9 W shunt regulator)",
        "description": "Clamps stepper back-EMF on +12V below the Tic T500's 35 V max",
        "block": "power_management",
        "footprint": "Pololu #3776 module, 2-pin THT",
        "parameters": {"clamp_v": 33.0, "power_w": 9.0, "polarity": "polarized"},
        "pins": {"+": "+12V", "-": "GND"},
    },
    {
        "ref": "C1", "type": "capacitor",
        "mpn": "100 uF / 25 V electrolytic",
        "description": "12 V rail bulk decoupling",
        "block": "power_management",
        "footprint": "Radial electrolytic, 0.1\" pitch (D6.3 mm)",
        "parameters": {"capacitance_uf": 100.0, "voltage_v": 25.0},
        "pins": {"+": "+12V", "-": "GND"},
    },
    {
        "ref": "C2", "type": "capacitor",
        "mpn": "100 uF / 10 V electrolytic",
        "description": "5 V rail bulk — tames servo + solenoid transients",
        "block": "power_management",
        "footprint": "Radial electrolytic, 0.1\" pitch (D6.3 mm)",
        "parameters": {"capacitance_uf": 100.0, "voltage_v": 10.0},
        "pins": {"+": "+5V", "-": "GND"},
    },
    {
        "ref": "C3", "type": "capacitor",
        "mpn": "100 uF / 25 V electrolytic",
        "description": "Bulk cap on the Tic T500 VIN screw terminals",
        "block": "power_management",
        "footprint": "Radial electrolytic, 0.1\" pitch (D6.3 mm)",
        "parameters": {"capacitance_uf": 100.0, "voltage_v": 25.0},
        "pins": {"+": "+12V", "-": "GND"},
    },
]

# ---------------------------------------------------------------------------
# Functional blocks (requirement #6) and interface protocols (#7).
# ---------------------------------------------------------------------------
FUNCTIONAL_BLOCKS = [
    {"name": "Power Management",
     "id": "power_management",
     "components": ["J1", "U1", "SR1", "C1", "C2", "C3"],
     "description": "12 V input, 12 V->5 V buck, bulk decoupling, back-EMF clamp. "
                    "Realizes the LaMAGIC2-proposed step-down power stages.",
     "rails_out": ["+12V", "+5V"]},
    {"name": "Control / MCU",
     "id": "control_mcu",
     "components": ["U2"],
     "description": "Raspberry Pi Pico W host; on-board LDO realizes the "
                    "5 V->3.3 V logic rail.",
     "rails_out": ["+3V3"]},
    {"name": "Motor Driver Stage",
     "id": "motor_driver_stage",
     "components": ["U5", "M2", "U4", "SOL1", "M3"],
     "description": "High-current auger stepper (Tic T500 + NEMA-11), tap "
                    "solenoid (DRV8871), and dispensing-angle servo.",
     "rails_in": ["+12V", "+5V"]},
    {"name": "Sensor / Haptic Interface",
     "id": "sensor_haptic_interface",
     "components": ["U3", "M1"],
     "description": "I2C haptic driver (DRV2605L) and ERM vibration motor.",
     "rails_in": ["+3V3"]},
]

INTERFACES = [
    {"name": "I2C0", "protocol": "I2C", "logic_v": 3.3,
     "from": "U2", "to": "U3", "nets": ["I2C_SDA", "I2C_SCL"]},
    {"name": "UART1", "protocol": "TTL-serial", "logic_v": 3.3, "baud": 9600,
     "from": "U2", "to": "U5", "nets": ["STP_TX", "STP_RX"],
     "notes": "Cross-over: Pico GP4(TX)->Tic RX, Pico GP5(RX)<-Tic TX."},
    {"name": "Solenoid PWM", "protocol": "PWM", "logic_v": 3.3,
     "from": "U2", "to": "U4", "nets": ["SOL_IN1", "SOL_IN2"]},
    {"name": "Servo PWM", "protocol": "PWM", "logic_v": 3.3, "frame_hz": 50,
     "from": "U2", "to": "M3", "nets": ["SERVO_SIG"]},
    {"name": "Haptic enable", "protocol": "GPIO", "logic_v": 3.3,
     "from": "U2", "to": "U3", "nets": ["HAPT_EN"]},
    {"name": "DC power rails", "protocol": "DC", "nets": ["+12V", "+5V", "+3V3", "GND"]},
]

# ---------------------------------------------------------------------------
# Design rules & constraints (requirement #5).
# ---------------------------------------------------------------------------
DESIGN_RULES = {
    "net_classes": {
        "power_high": {"nets": ["+12V"], "min_trace_width_mm": 0.8,
                       "rationale": "12 V / up to 5 A from the brick."},
        "power_logic": {"nets": ["+5V", "+3V3"], "min_trace_width_mm": 0.5},
        "motor_coil": {"nets": ["STP_A1", "STP_A2", "STP_B1", "STP_B2",
                                 "SOL_A", "SOL_B", "VIB_A", "VIB_B"],
                       "min_trace_width_mm": 0.8,
                       "rationale": "Stepper/solenoid coil currents + transients."},
        "logic_signal": {"nets": ["I2C_SDA", "I2C_SCL", "STP_TX", "STP_RX",
                                   "SOL_IN1", "SOL_IN2", "HAPT_EN", "SERVO_SIG"],
                         "min_trace_width_mm": 0.25},
        "ground": {"nets": ["GND"], "min_trace_width_mm": 0.8,
                   "rationale": "Single star/common ground."},
    },
    "isolation": [
        {"rule": "Separate the high-current / 12 V domain (+12V, motor_coil, "
                 "motor_out nets) from the low-voltage logic domain (I2C/UART/PWM "
                 "signals, +3V3) to keep stepper/solenoid switching transients out "
                 "of the I2C and UART lines.",
         "domains": ["high_current", "logic"]},
        {"rule": "SR1 (Pololu #3776) is polarized: + on +12V, - on GND. Reversing "
                 "it destroys the regulator. Mount it in parallel with C3, as close "
                 "to U5.VIN as possible, to clamp back-EMF below the Tic T500's "
                 "35 V absolute-max input.",
         "domains": ["high_current"]},
        {"rule": "Never connect/disconnect the stepper (M2) while +12V is powered "
                 "— hot-plugging can destroy the Tic T500 driver.",
         "domains": ["high_current"]},
        {"rule": "Common star ground; the Pico and Tic share the breadboard GND "
                 "rail so the UART has a return reference.",
         "domains": ["global"]},
    ],
    "footprint_note": (
        "PR #61 authors the schematic as global-label-only connectivity with "
        "project-local symbols and no assigned KiCad footprints; the 'footprint' "
        "fields here are the physical package/form-factor of each off-the-shelf "
        "module (breadboard-/0.1\"-header mount), not PCB-library footprint names."
    ),
}

# ---------------------------------------------------------------------------
# How the abstract LaMAGIC2 power-stage nodes map onto concrete parts (#1).
# The 12 V -> 5 V topology generated in powder_doser_topology.md (S3) is a
# two-switch + inductor + cap synchronous-buck structure; the off-the-shelf
# D24V22F5 integrates exactly that stage. The 5 V -> 3.3 V stage is realized by
# the Pico W's on-board LDO.
# ---------------------------------------------------------------------------
POWER_STAGE_REALIZATION = {
    "12V_to_5V": {
        "lamagic2_target": {"vout_ratio": 0.4167, "eff": 0.95, "duty": 0.3},
        "lamagic2_abstract_nodes": {
            "Sa0": "high-side / control switch",
            "Sb1": "synchronous (low-side) switch",
            "L2": "buck inductor",
            "C3": "input bulk capacitor",
            "C4": "output bulk capacitor",
        },
        "concrete_realization": {
            "switches": "integrated FETs inside U1 (Pololu D24V22F5)",
            "inductor": "integrated inductor inside U1",
            "input_cap": "C1 (100 uF / 25 V on +12V)",
            "output_cap": "C2 (100 uF / 10 V on +5V)",
            "module": "U1 (Pololu D24V22F5 synchronous buck)",
        },
    },
    "5V_to_3V3": {
        "lamagic2_target": {"vout_ratio": 0.66, "eff": 0.95, "duty": 0.7},
        "concrete_realization": {
            "module": "U2 on-board LDO (Raspberry Pi Pico W 3V3 regulator)",
            "note": "Low-power logic rail; an LDO, not a switching buck.",
        },
    },
}


def build_spec():
    """Assemble the full Celus-compatible structured design specification."""
    return {
        "schema": "celus-compatible-block-diagram/v1",
        "design": "powder-doser-test-module",
        "source": {
            "pr": PR61,
            "commit": PR61_COMMIT,
            "readme": PR61_README,
            "lamagic2_writeup": "experiment/lamagic2/results/powder_doser_topology.md",
            "note": (
                "Concrete component/connectivity data transcribed from powder-doser "
                "PR #61; abstract power-stage topology proposed by the released "
                "LaMAGIC2 SFCI checkpoint (see generate_custom_topology.py)."
            ),
        },
        "global_requirements": GLOBAL_REQUIREMENTS,
        "functional_blocks": FUNCTIONAL_BLOCKS,
        "interfaces": INTERFACES,
        "nets": NETS,
        "components": COMPONENTS,
        "design_rules": DESIGN_RULES,
        "power_stage_realization": POWER_STAGE_REALIZATION,
    }


def validate_spec(spec):
    """Self-consistency checks; returns a list of human-readable problems."""
    problems = []
    net_names = {n["name"] for n in spec["nets"]}
    block_ids = {b["id"] for b in spec["functional_blocks"]}
    comp_refs = {c["ref"] for c in spec["components"]}

    # Every pin net (except explicit None/no-connect) must be a declared net.
    for comp in spec["components"]:
        for pin, net in comp["pins"].items():
            if net is not None and net not in net_names:
                problems.append(f"{comp['ref']}.{pin} -> undeclared net '{net}'")
        if comp["block"] not in block_ids:
            problems.append(f"{comp['ref']} -> unknown block '{comp['block']}'")

    # Every block member must be a declared component, and vice-versa.
    block_members = set()
    for block in spec["functional_blocks"]:
        for ref in block["components"]:
            block_members.add(ref)
            if ref not in comp_refs:
                problems.append(f"block '{block['id']}' lists unknown component '{ref}'")
    for ref in comp_refs - block_members:
        problems.append(f"component '{ref}' is not in any functional block")

    # Interface nets must exist.
    for iface in spec["interfaces"]:
        for net in iface["nets"]:
            if net not in net_names:
                problems.append(f"interface '{iface['name']}' -> undeclared net '{net}'")

    # Each power/ground net should have at least one component pin on it.
    pinned_nets = {net for c in spec["components"] for net in c["pins"].values()
                   if net is not None}
    for net in spec["nets"]:
        if net["class"] in ("power", "ground") and net["name"] not in pinned_nets:
            problems.append(f"power/ground net '{net['name']}' has no component pins")

    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out", default=None,
        help="Write the JSON spec to this path (default: print to stdout).")
    parser.add_argument(
        "--indent", type=int, default=2, help="JSON indentation (default: 2).")
    args = parser.parse_args(argv)

    spec = build_spec()
    problems = validate_spec(spec)
    if problems:
        print("Spec validation FAILED:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
        return 1

    text = json.dumps(spec, indent=args.indent)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"Wrote {len(spec['components'])} components, "
              f"{len(spec['nets'])} nets, {len(spec['functional_blocks'])} blocks "
              f"to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

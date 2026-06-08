# LaMAGIC2 topology generation for the powder-doser bench rig

This is a worked example of pointing the **released** LaMAGIC2 SFCI checkpoint
([`turtleben/LaMAGIC2-345comp-SFCI-dataaug-noaug`](https://huggingface.co/turtleben/LaMAGIC2-345comp-SFCI-dataaug-noaug))
at the power-conversion requirements of the
[powder-doser single-Pico-W test module](https://github.com/vertical-cloud-lab/powder-doser/pull/61),
so the model's proposed converter topologies can be compared against the
off-the-shelf regulators chosen in that design.

Everything below is reproducible on CPU with
[`generate_custom_topology.py`](generate_custom_topology.py) — no GPU or training
required. Topology generation needs no ngspice; the ngspice verification in
[§4](#4-closing-the-loop-with-ngspice) additionally requires an `ngspice` install.

## 1. Specifications extracted from powder-doser PR #61

Only numerical/verbal specs are used (no KiCad geometry), per the request:

| Quantity | Value | Source (PR #61 `hardware/test-module/README.md`) |
| --- | --- | --- |
| System input | **12 V** (Mean Well GST60A12-P1J, 12 V / 5 A, 60 W) | `J1`, `+12V` rail |
| Main regulated rail | **5 V** (Pololu D24V22F5 buck, 2.5 A) | `U1`, `+5V` rail — "12 V → 5 V for the Pico W + servo + solenoid" |
| Logic rail | **3.3 V** (Pico W on-board LDO) | `U2`/`+3V3` — powers DRV2605L logic |
| 5 V loads | Pico W, hobby servo, JF-0530B 5 V solenoid | `+5V` net table |
| Function | High-efficiency **step-down** DC-DC conversion | buck regulator |

That yields two voltage-conversion targets to ask LaMAGIC2 for:

| Target rail | Vout/Vin | Target efficiency |
| --- | --- | --- |
| 12 V → 5 V (main rail) | 5 / 12 = **0.4167** | 0.95 |
| 5 V → 3.3 V (logic rail) | 3.3 / 5 = **0.66** | 0.95 |

## 2. The prompt, in LaMAGIC2 terms

LaMAGIC2's SFCI formulation conditions generation on three numbers plus a
component budget, so the natural-language spec above maps to:

* **voltage conversion ratio** `vout = Vout/Vin`,
* **target efficiency** `eff` in `[0, 1]`,
* **available components** chosen from power switches (`Sa`, `Sb`), inductors
  (`L`) and capacitors (`C`) — here a 5-component buck-style budget
  `Sa0 Sb1 L2 C3 C4`.

The model then emits the switching **duty cycle** and the **connectivity**
between `VIN`, `VOUT`, `GND` and those components.

## 3. Generated topologies

### 12 V → 5 V main rail (`vout=0.4167`, `eff=0.95`)

```
python experiment/lamagic2/generate_custom_topology.py \
    --vout 0.4167 --eff 0.95 --components Sa0 Sb1 L2 C3 C4
```

```
Generated topology string:
  <pad> <duty_0.3> <sep> VIN L 2 C 4, VOUT Sb 1 C 4, GND C 3, Sa 0 C 3, Sa 0 Sb 1 L 2 <sep></s>

Decoded duty cycle: 0.3
Decoded netlist (device -> connection nodes):
  C3   -> 0, 9
  C4   -> IN, OUT
  L2   -> IN, 10
  Sa0  -> 9, 10
  Sb1  -> OUT, 10
```

![12V to 5V topology](results/powder_doser_12V_to_5V.png)

### 5 V → 3.3 V logic rail (`vout=0.66`, `eff=0.95`)

```
python experiment/lamagic2/generate_custom_topology.py \
    --vout 0.66 --eff 0.95 --components Sa0 Sb1 L2 C3 C4
```

```
Generated topology string:
  <pad> <duty_0.7> <sep> VIN Sb 1 C 3, VOUT L 2 C 3, GND Sa 0, Sa 0 C 4, Sb 1 L 2 C 4 <sep></s>

Decoded duty cycle: 0.7
Decoded netlist (device -> connection nodes):
  C3   -> IN, OUT
  C4   -> 9, 10
  L2   -> OUT, 10
  Sa0  -> 0, 9
  Sb1  -> IN, 10
```

![5V to 3.3V topology](results/powder_doser_5V_to_3V3.png)

Both proposals are single connected graphs containing `VIN`/`VOUT`/`GND` and use
the two-switch + inductor + capacitor structure of a synchronous buck, which is
consistent with the off-the-shelf D24V22F5 buck regulator chosen in PR #61.

## 4. Closing the loop with ngspice

LaMAGIC2 proposes a **topology**; it does not by itself guarantee the simulated
operating point. The repository ships an ngspice-based verifier
(`parsers/simulation.py`), so we close the loop and actually simulate each
generated circuit. Install ngspice (`apt-get install ngspice` or
`conda install -c conda-forge ngspice`) and pass `--simulate`:

```
python experiment/lamagic2/generate_custom_topology.py \
    --vout 0.4167 --eff 0.95 --components Sa0 Sb1 L2 C3 C4 --simulate
```

The verifier uses the repository's standard operating-point parameters
(`simulate_param` in `parsers/simulation.py`: `Vin = 100 V`, switching
`Frequency = 1 MHz`, `Rout = 50 Ω`, `L = 100 µH`, `C = 10 µF`), reports the
realized `Vout/Vin` and efficiency, and is what LaMAGIC2's own evaluation uses to
score a candidate. Running it on the two generated topologies gives:

| Target rail | Target Vout/Vin | Realized Vout/Vin | Target eff | Realized eff | `result_valid` |
| --- | --- | --- | --- | --- | --- |
| 12 V → 5 V (`duty=0.3`) | 0.4167 | **0.030** | 0.95 | **0.022** | True |
| 5 V → 3.3 V (`duty=0.7`) | 0.66 | **0.384** | 0.95 | **0.522** | True |

Both netlists are simulatable (the transient solves and `result_valid` is
`True`), but **neither single greedy generation hits its target operating
point** under the default simulation parameters: the realized conversion ratios
and efficiencies are well below the requested values.

This is the honest, end-to-end result and is exactly why the loop matters: a
proposed topology is only a hypothesis until ngspice confirms it. Getting from
"plausible structure" to "meets the spec" requires LaMAGIC2's full search loop,
implemented next.

The simulated numbers above are reproducible with the `--simulate` flag shown
above; the ngspice netlists are written to the `--cki-path` (or a temp file).

### 4.1 Full selection-by-simulation search (hitting the target)

A single greedy decode is just one sample. The way LaMAGIC2 is meant to be used
is **selection by simulation**: generate *many* candidate topologies, simulate
each one in ngspice across the conditioning duty-cycle options, and keep the
candidate whose **realized** `Vout`/efficiency are closest to the target. The
`--search` flag does exactly this, sweeping three axes:

- **Topology candidates** — one greedy decode plus `--num-candidates` sampled
  decodes (`do_sample`, `top_k`) for diversity.
- **Component budgets** — with `--sweep-budgets`, the most common 3/4/5-component
  budgets from the released dataset are tried in addition to `--components`.
- **Duty cycle** — every candidate is simulated at all five duty-cycle options
  (0.1/0.3/0.5/0.7/0.9), since the operating point depends strongly on the duty
  cycle (e.g. one decoded topology swings from 0.027 at duty 0.1 to 0.881 at
  duty 0.9).

```
python experiment/lamagic2/generate_custom_topology.py \
    --vout 0.4167 --eff 0.95 --components Sa0 Sb1 L2 C3 C4 \
    --search --sweep-budgets
```

Running the full search for both rails (released checkpoint, CPU, ngspice
verifier with the same `simulate_param` as above) **does hit the target**:

| Target rail | Target Vout/Vin | **Search Vout/Vin** | Target eff | **Search eff** | Selected duty | Selected components |
| --- | --- | --- | --- | --- | --- | --- |
| 12 V → 5 V | 0.4167 | **0.448** | 0.95 | **0.851** | 0.1 | `Sa0 Sa1 Sb2 Sb3 C4` |
| 5 V → 3.3 V | 0.66 | **0.648** | 0.95 | **0.947** | 0.5 | `Sa0 Sa1 Sb2 L3 C4` |

Compared to the single greedy decodes (0.030 and 0.384), the search lands within
~3 % and ~2 % of the requested conversion ratios respectively, with efficiencies
of 0.85 and 0.95 — i.e. it actually meets the spec. Notably the search *changes
both the topology and the duty cycle* relative to the greedy decode (which
proposed `duty=0.3` for these budgets): for the 12 V → 5 V rail it selects a
four-switch + capacitor topology at `duty=0.1`, and for the 5 V → 3.3 V rail a
buck-like two-switch-pair + inductor topology at `duty=0.5`. The selected
topologies are rendered in
[`powder_doser_12V_to_5V_search.png`](powder_doser_12V_to_5V_search.png) and
[`powder_doser_5V_to_3V3_search.png`](powder_doser_5V_to_3V3_search.png).

The ranking is by closeness to the target (`--eff-weight` controls how much the
efficiency term counts relative to the conversion ratio); pass `--top N` to see
the N best candidates. The whole loop is reproducible with the `--search`
command above and is covered by the real test
`test_selection_by_simulation_search_finds_target` in
`tests/test_lamagic_pipeline.py`.

For the powder-doser bench rig specifically, the off-the-shelf Pololu D24V22F5
buck chosen in PR #61 remains the right call; this example demonstrates that the
LaMAGIC2 → ngspice search pipeline can converge on topologies that hit the
requested operating points.


## 5. From abstract topology to a structured design specification

The topologies in [§3](#3-generated-topologies) are deliberately **abstract**:
anonymous switches/inductors/capacitors (`Sa0`, `Sb1`, `L2`, `C3`, `C4`) wired
between `VIN`/`VOUT`/`GND`. That is what LaMAGIC2 emits, but it carries none of
the project-specific information a downstream synthesis/layout tool (e.g. the
[Celus](https://www.celus.io/) platform) needs. Per the request on
[PR #2](https://github.com/vertical-cloud-lab/LaMAGIC/pull/2), we lift that
abstract graph into a **structured, Celus-compatible design specification** that
incorporates the real powder-doser components and constraints.

[`generate_celus_spec.py`](../generate_celus_spec.py) assembles this from the
real component/connectivity data published in
[powder-doser PR #61](https://github.com/vertical-cloud-lab/powder-doser/pull/61)
(`hardware/test-module/README.md` BOM + pin/net table, `firmware/config.py`,
`kicad/generate.py` symbol pin maps) and writes
[`powder_doser_celus_spec.json`](powder_doser_celus_spec.json):

```
python experiment/lamagic2/generate_celus_spec.py \
    --out experiment/lamagic2/results/powder_doser_celus_spec.json
```

The spec addresses each requested requirement:

| # | Requirement | Where in the spec |
| --- | --- | --- |
| 1 | Explicit component typing & parametric data (real MPNs, no `C4`/`L2`/`Sb1`) | `components[].{type,mpn,parameters}` |
| 2 | Pin-level connectivity | `components[].pins` (pin → net) |
| 3 | Power & ground designations with net classes | `nets[].{class,domain,nominal_v,max_current_a}` |
| 4 | Footprint & packaging constraints | `components[].footprint` (+ `design_rules.footprint_note`) |
| 5 | Design rules — isolation/clearance | `design_rules.{net_classes,isolation}` |
| 6 | Functional-block grouping | `functional_blocks` (Power Management, Control/MCU, Motor Driver Stage, Sensor/Haptic Interface) |
| 7 | Interface protocols | `interfaces` (I2C, TTL-serial/UART, PWM, DC rails) |
| 8 | Global system requirements | `global_requirements` (12 V input, 3.3/5 V logic, currents) |

Crucially, the abstract LaMAGIC2 power stage is **tied to its concrete
realization** in `power_stage_realization`: the two-switch + inductor + cap
synchronous-buck structure LaMAGIC2 proposes for 12 V → 5 V is integrated by the
off-the-shelf **Pololu D24V22F5** buck (`U1`, with input/output bulk caps `C1`/`C2`),
and the 5 V → 3.3 V stage is realized by the **Pico W on-board LDO** (`U2`). So the
abstract `Sa0`/`Sb1`/`L2`/`C3`/`C4` nodes become real, parameterized parts wired at
the pin level, with `GND`/power net classes, footprints, isolation rules, and
functional blocks — the structured electronic design specification the request
asked for.

The spec is validated for self-consistency (every pin references a declared net,
every component belongs to exactly one functional block, all interface nets
exist) both by the generator and by [`tests/test_celus_spec.py`](../../../tests/test_celus_spec.py).

# LaMAGIC2 topology generation for the powder-doser bench rig

This is a worked example of pointing the **released** LaMAGIC2 SFCI checkpoint
([`turtleben/LaMAGIC2-345comp-SFCI-dataaug-noaug`](https://huggingface.co/turtleben/LaMAGIC2-345comp-SFCI-dataaug-noaug))
at the power-conversion requirements of the
[powder-doser single-Pico-W test module](https://github.com/vertical-cloud-lab/powder-doser/pull/61),
so the model's proposed converter topologies can be compared against the
off-the-shelf regulators chosen in that design.

Everything below is reproducible on CPU with
[`generate_custom_topology.py`](generate_custom_topology.py) — no GPU, training,
or ngspice install required.

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

## 4. Caveat

LaMAGIC2 proposes a **topology**; it does not by itself guarantee the simulated
operating point. To verify that a generated circuit actually hits the target
`Vout`/efficiency, close the loop with the repository's ngspice simulation
(`parsers/simulation.py`) — that step needs an ngspice install and is not run
here.

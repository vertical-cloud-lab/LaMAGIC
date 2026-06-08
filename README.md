# LaMAGIC and LaMAGIC2: Language-Model-based Topology Generation for Analog Integrated Circuits

### Published at ICML 2024 & ICML 2025

This repository provides the official implementation of LaMAGIC and LaMAGIC2, two approaches leveraging Language Models (LMs) for automated topology generation of analog integrated circuits.

* **LaMAGIC** ([ICML'24 Paper](https://arxiv.org/pdf/2407.18269))
* **LaMAGIC2** ([ICML'25 Paper](https://arxiv.org/abs/2506.10235))

---

## Installation

### Environment Setup

First, create a Conda environment using:

```bash
conda env create -f environment.yml
```

---

## Dataset and Model Setup

### Base Model

Download the pretrained `flan-T5-base` model from [Hugging Face](https://huggingface.co/google/flan-t5-base). Update the model path in all YAML configuration files located in:

```
analog_LLM/configs
```

### Dataset

Clone the dataset from [LaMAGIC-dataset](https://huggingface.co/datasets/turtleben/LaMAGIC-dataset).

Target data is located at:

```
[your_save_path]/LaMAGIC-dataset/transformed
```

* For LaMAGIC2 SOTA experiments, use:

  * `LaMAGIC2/SFCI_345comp.json`
  * `LaMAGIC2/SFCI_6comp.json`

---

## Training

Training scripts are organized by each paper under the `experiment` directory.

### LaMAGIC (ICML'24)

#### Initial training on 3, 4, and 5-component circuits:

* **Naïve (NF), Canonical (CF), Canonical + Duty Cycle (CFDC)**:

```bash
python experiment/lamagic1/trn_LLM_instruction.py
```

* **Pure-text adjacency-matrix (PM)**:

```bash
python experiment/lamagic1/trn_LLM_pure_text_matrix_form.py
```

* **Float-input adjacency-matrix (FM)** *(core contribution)*:

```bash
python experiment/lamagic1/trn_LLM_float_input_matrix_form.py
```

#### Fine-tuning on 6-component circuits (limited data: 500, 1000, 2000 samples):

```bash
python experiment/lamagic1/trn_LLM_6_comp.py
```

---

### LaMAGIC2 (ICML'25)

#### Initial training on 3, 4, and 5-component circuits:

* **Succinct Float-input Matrix (SFM), Succinct Float-input Canonical with Identifier (SFCI)** *(core contribution)*:

```bash
python experiment/lamagic2/trn_pure_tranformer.py
```

#### Fine-tuning on 6-component circuits (limited data: 500, 1000, 2000 samples):

```bash
python experiment/lamagic2/trn_pure_tranformer_6comp.py
```

#### Released Model Checkpoints:

* **3, 4, 5-component circuits (SFM):**

  1. Trained on SFM with data augmentation (vertex random shuffle): [LaMAGIC2-345comp-SFM-dataaug](https://huggingface.co/turtleben/LaMAGIC2-345comp-SFM-dataaug)
  2. After data augment, trained on SFM without data augmentation: [LaMAGIC2-345comp-SFM-dataaug-noaug](https://huggingface.co/turtleben/LaMAGIC2-345comp-SFM-dataaug-noaug)

* **3, 4, 5-component circuits (SFCI):**

  1. Trained on SFCI with data augmentation (vertex random shuffle): [LaMAGIC2-345comp-SFCI-dataaug](https://huggingface.co/turtleben/LaMAGIC2-345comp-SFCI-dataaug)
  2. After data augment, trained on SFCI without data augmentation: [LaMAGIC2-345comp-SFCI-dataaug-noaug](https://huggingface.co/turtleben/LaMAGIC2-345comp-SFCI-dataaug-noaug)

* **6-component circuits (SFM):**

  * [LaMAGIC2-6Comp-SFM-dnum500](https://huggingface.co/turtleben/LaMAGIC2-6Comp-SFM-dnum500)
  * [LaMAGIC2-6Comp-SFM-dnum1000](https://huggingface.co/turtleben/LaMAGIC2-6Comp-SFM-dnum1000)
  * [LaMAGIC2-6Comp-SFM-dnum2000](https://huggingface.co/turtleben/LaMAGIC2-6Comp-SFM-dnum2000)

* **6-component circuits (SFCI):**

  * [LaMAGIC2-6Comp-SFCI-dnum500](https://huggingface.co/turtleben/LaMAGIC2-6Comp-SFCI-dnum500)
  * [LaMAGIC2-6Comp-SFCI-dnum1000](https://huggingface.co/turtleben/LaMAGIC2-6Comp-SFCI-dnum1000)
  * [LaMAGIC2-6Comp-SFCI-dnum2000](https://huggingface.co/turtleben/LaMAGIC2-6Comp-SFCI-dnum2000)

---

## Generating a topology for a custom target

To ask the released SFCI checkpoint for a converter that hits an arbitrary
target (instead of sweeping the released dataset), use
`experiment/lamagic2/generate_custom_topology.py`. It runs the real model
end-to-end on CPU and prints/draws the generated topology:

```bash
python experiment/lamagic2/generate_custom_topology.py \
    --vout 0.4167 --eff 0.95 --components Sa0 Sb1 L2 C3 C4
```

`--vout` is the target voltage conversion ratio `Vout/Vin`, `--eff` the target
efficiency, and `--components` the available switches/inductors/capacitors. A
worked example mapping the [powder-doser bench rig](https://github.com/vertical-cloud-lab/powder-doser/pull/61)
power rails (12 V → 5 V and 5 V → 3.3 V) onto LaMAGIC2 is in
[`experiment/lamagic2/results/powder_doser_topology.md`](experiment/lamagic2/results/powder_doser_topology.md).

---

## Testing

The `tests/` directory contains a basic test suite that can be used to
sanity-check the repository. It has two layers:

* **Unit tests** (`tests/test_topo_graph.py`) for the self-contained graph
  utilities in `topo_data_util` — these only need `numpy` + `pytest`.
* **End-to-end tests** (`tests/test_lamagic_pipeline.py`) that run the *real*
  pipeline against the *real* released artifacts: they download the
  [LaMAGIC2 `SFCI_345comp` dataset](https://huggingface.co/datasets/turtleben/LaMAGIC-dataset)
  and the `google/flan-t5-base` tokenizer/config, then
  1. decode real SFCI formulation strings back into circuit netlists/graphs via
     `parsers/simulation.py`, asserting the recovered devices and duty cycle
     match the dataset, and
  2. run a real forward/backward training step of the custom encoder-decoder
     transformer in `analog_LLM/models/T5_transformer.py` (with the float
     `vout`/`eff`/duty-cycle prefixes), asserting a finite loss and gradients.

Install the test requirements and run:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU build is fine
pip install -r requirements-test.txt
pytest tests/
```

The end-to-end tests require network access to the Hugging Face Hub. Both the
dataset (`turtleben/LaMAGIC-dataset`) and the `google/flan-t5-base` artifacts are
public, so **no Hugging Face token/API key is required** — the tests download them
directly and fail if the Hub is unreachable. No GPU, trained checkpoint, or
ngspice install is required.

---

## Citation

If you use this work in your research, please cite our papers:

```bibtex
@InProceedings{chang2024lamagic,
  title = 	 {{L}a{MAGIC}: Language-Model-based Topology Generation for Analog Integrated Circuits},
  author =       {Chang, Chen-Chia and Shen, Yikang and Fan, Shaoze and Li, Jing and Zhang, Shun and Cao, Ningyuan and Chen, Yiran and Zhang, Xin},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {6253--6262},
  year = 	 {2024},
  month = 	 {21--27 Jul},
  organization =    {PMLR},
}

@inproceedings{chang2025lamagic2,
  title={{L}a{MAGIC}2: Advanced Circuit Formulations for Language Model-Based Analog Topology Generation},
  author={Chang, Chen-Chia and Lin, Wan-Hsuan and Shen, Yikang and Chen, Yiran and Zhang, Xin},
  booktitle={Proceedings of the 42st International Conference on Machine Learning},
  year={2025},
  organization =    {PMLR},
}
```

---

## Contact

For questions or further collaboration, please reach out to the authors.

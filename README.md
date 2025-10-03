---

# ComfyUI Switch Samplers Pack

This custom node pack for **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** introduces advanced sampler and scheduler switching strategies. It allows users to dynamically change samplers, schedulers, models, VAEs, CFG scales, denoise values, and conditioning between stages of a generation.

The pack is designed for both **same-family checkpoints** (e.g. SDXL variants) and **cross-family workflows** (e.g. SDXL → Flux → Qwen), with automatic latent bridging via VAE decode/encode.

---

## ✨ Features

* **12+ Fully Working Sampler Nodes**

  * Step Switch KSampler
  * MultiStep KSampler
  * Stochastic Switch Sampler
  * Frequency Switch Sampler
  * Random Switch Sampler
  * Conditional Switch Sampler
  * Time-Based Switch Sampler
  * Gradient Switch Sampler
  * Pattern Switch Sampler
  * Threshold Switch Sampler
  * Adaptive Switch Sampler
  * Sequential Switch Sampler

* **Upgraded Cross-Model Nodes**

  * **CrossStepSwitchKSampler** → Two-stage sampler with optional second model/vae/text encoder. Perfect for workflows like *SDXL → Flux*.
  * **CrossMultiStepKSampler** → Three-stage sampler with full control over models, VAEs, CFGs, denoise values, samplers, schedulers, and conditioning for each stage.

* **Flexible Conditioning**

  * Each stage can use its own **positive** and **negative** conditioning (from the correct text encoder for the model in that stage).
  * Falls back to the previous stage if not connected.

* **Latent Bridging**

  * Automatically performs **VAE decode → encode** when switching between different model/vae pairs.
  * Ensures compatibility when mixing architectures.

---

## 📦 Installation

1. Clone or download this repository into your ComfyUI `custom_nodes` folder:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/YOUR-USERNAME/comfyui-switch-samplers.git
   ```

   Or manually place the folder inside `ComfyUI/custom_nodes`.

2. Restart ComfyUI.

3. The new nodes will appear under:

   * **Sampling / Switch Samplers** (all original switching nodes)
   * **Sampling / Cross Samplers** (the new cross-model nodes)

---

## 🛠️ Usage

* Connect a **Model**, **VAE**, **Conditioning** (positive/negative), and **Latent Image** into one of the switch nodes.
* Select samplers, schedulers, step counts, CFG values, and denoise levels per stage.
* For cross-model nodes, connect multiple **models/VAEs/conditionings** to chain different architectures in one workflow.
* Outputs a new **LATENT** tensor for further processing or decoding.

---

## ⚡ Example Workflows

* **StepSwitchKSampler**
  Run first 10 steps with Euler on SDXL, then switch to 10 steps with DPM++ 2M Karras for refinement.

* **MultiStepKSampler**
  Run 3 consecutive sampler stages with different samplers, schedulers, and denoise values.

* **CrossStepSwitchKSampler**
  Run SDXL for the first half, then seamlessly switch to Flux with its own VAE and text encoder for final refinement.

* **CrossMultiStepKSampler**
  Run three different models in sequence, e.g.:

  1. SDXL for base composition
  2. Flux for style transfer
  3. Qwen for final polish

---

## 📖 Node Categories

* **Sampling / Switch Samplers** → All single-model switch strategies.
* **Sampling / Cross Samplers** → Cross-model workflow samplers.

---

## 📜 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute it, with attribution.

---

## 🙌 Credits

* Built on top of [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
* Extended sampler logic inspired by community contributions.
* Cross-model VAE bridging and conditioning handling by this custom pack.

---

👉 That gives you a **clean GitHub README** that documents everything in the pack, including your two new upgraded nodes.

Do you also want me to draft a **`pyproject.toml`** (for PyPI-style packaging) that matches this README?

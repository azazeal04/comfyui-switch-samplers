---

# ComfyUI Switch Samplers Pack

This custom node pack for **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** introduces advanced sampler and scheduler switching strategies. It allows users to dynamically change samplers, schedulers, models, VAEs, CFG scales, denoise values, and conditioning between stages of a generation.

The pack is designed for both **same-family checkpoints** (e.g. SDXL variants).

---

## ✨ Features

* **2+ Fully Working Sampler Nodes**

  * Step Switch KSampler
  <img width="1587" height="955" alt="StepSwitchKSampler" src="https://github.com/user-attachments/assets/a4352b1f-10fa-4172-97ef-ef22501374b5" />

  
  * MultiStep KSampler
 <img width="1592" height="1007" alt="MultiStepKSampler" src="https://github.com/user-attachments/assets/cc7cfffc-e322-4fa3-a32a-0c5d204c7f25" />

    
 

* **Upgraded Cross-Model Nodes**
Still to be added when tests are completed.

---

## 📦 Installation

1. Clone or download this repository into your ComfyUI `custom_nodes` folder:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/azazeal04/comfyui-switch-samplers.git
   ```

   Or manually place the folder inside `ComfyUI/custom_nodes`.

2. Restart ComfyUI.

3. The new nodes will appear under:

   * **Azazeal / Switch Samplers** (all original switching nodes)
   
---

## 🛠️ Usage

* Connect a **Model**, **VAE**, **Conditioning** (positive/negative), and **Latent Image** into one of the switch nodes.
* Select samplers, schedulers, step counts, CFG values, and denoise levels per stage.
* Outputs a new **LATENT** tensor for further processing or decoding.

---

## ⚡ Example Workflows

* **StepSwitchKSampler**
  Run first 10 steps with Euler on SDXL, then switch to 10 steps with DPM++ 2M Karras for refinement.

* **MultiStepKSampler**
  Run 3 consecutive sampler stages with different samplers, schedulers, and denoise values.

---

## 📖 Node Categories

* **Azazeal / Switch Samplers** → Single-model switch strategies.

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


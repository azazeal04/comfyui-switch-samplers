---

# ComfyUI Switch Samplers Pack

This custom node pack for **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** introduces advanced sampler and scheduler switching strategies. It allows users to dynamically change samplers, schedulers, models, VAEs, CFG scales, denoise values, and conditioning between stages of a generation.

The pack is designed for both **same-family checkpoints** (e.g. SDXL variants).

---

## ✨ Features

* **4 Fully Working Sampler Nodes**

  * Step Switch KSampler
  <img width="1587" height="955" alt="StepSwitchKSampler" src="https://github.com/user-attachments/assets/a4352b1f-10fa-4172-97ef-ef22501374b5" />

  
  * MultiStep KSampler
 <img width="1592" height="1007" alt="MultiStepKSampler" src="https://github.com/user-attachments/assets/cc7cfffc-e322-4fa3-a32a-0c5d204c7f25" />

    
 

* **Upgraded Cross-Model Nodes**

  * **CrossStepSwitchKSampler** → Two-stage sampler with optional second model/vae/text encoder. Perfect for workflows like *SDXL → Flux*.
 <img width="1637" height="1275" alt="CrossStepSwitchKSampler" src="https://github.com/user-attachments/assets/07e04b99-4d55-4536-8805-d826526b64e6" />

    
  * **CrossMultiStepKSampler** → Three-stage sampler with full control over models, VAEs, CFGs, denoise values, samplers, schedulers, and conditioning for each stage.
 <img width="1764" height="1686" alt="CrossMultiStepKSampler" src="https://github.com/user-attachments/assets/cf709a04-8796-4352-be3a-e5aa175f640c" />

    

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

* **CrossStepSwitchKSampler**
  Run SDXL for the first half, then seamlessly switch to Flux with its own VAE and text encoder for final refinement.

* **CrossMultiStepKSampler**
  Run three different models in sequence, e.g.:

  1. SDXL for base composition
  2. Flux for style transfer
  3. Qwen for final polish

* **Important Note**
  If you want to use Flux, Qwen or any other model on the StepSwitchKSampler and MultiStepKSampler nodes or as first models on the StepSwitchKSampler and MultiStepKSampler you will first need to encode the empty latent with SDXL VAE and then decode it with VAE of the     model you want to use before the Ksampler.
  ![Screenshot_1834](https://github.com/user-attachments/assets/34b31971-da0d-4afb-8150-5063af919006)

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


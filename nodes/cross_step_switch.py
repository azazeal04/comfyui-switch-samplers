import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.samplers as cs
from .helpers import _call_ksampler, _make_node_class

def _cross_step_switch_handler(model, latent, kwargs):
    switch_point = kwargs.get("switch_point", 10)
    total_steps  = kwargs.get("total_steps", 20)

    sampler_before = kwargs.get("sampler_before", cs.KSampler.SAMPLERS[0])
    sampler_after  = kwargs.get("sampler_after", cs.KSampler.SAMPLERS[0])

    scheduler_before = kwargs.get("scheduler_before", cs.KSampler.SCHEDULERS[0])
    scheduler_after  = kwargs.get("scheduler_after", cs.KSampler.SCHEDULERS[0])

    cfg_before = kwargs.get("cfg_before", 7.5)
    cfg_after  = kwargs.get("cfg_after", 7.5)

    denoise_before = kwargs.get("denoise_before", 1.0)
    denoise_after  = kwargs.get("denoise_after", 1.0)

    seed = kwargs.get("seed", 0)

    m1 = kwargs.get("model1")
    m2 = kwargs.get("model2", m1)

    vae1 = kwargs.get("vae1")
    vae2 = kwargs.get("vae2", vae1)

    pos1 = kwargs.get("positive1", kwargs.get("positive"))
    neg1 = kwargs.get("negative1", kwargs.get("negative"))
    pos2 = kwargs.get("positive2", pos1)
    neg2 = kwargs.get("negative2", neg1)

    device = mm.get_torch_device()

    # --- FIRST SAMPLING ---
    out1 = _call_ksampler(m1, latent, switch_point, sampler_before, scheduler_before,
                         cfg_before, pos1, neg1, seed, denoise_before)

# ---- Decode and re-encode between models ----
    if vae1 is not None and vae2 is not None and vae1 != vae2:
        img = vae1.decode(out1["samples"])
        if img is None:
            raise RuntimeError("VAE1 decode returned None — cannot continue switch.")
        
         # ensure tensor
        img = img if torch.is_tensor(img) else torch.tensor(img)

        # handle both 4D and 5D (video-like) outputs
        if img.dim() == 5:
            # merge temporal dimension for encoding into 2D vae
            b, f, h, w, c = img.shape
            img = img.view(b * f, h, w, c)
        elif img.dim() == 3:
            img = img.unsqueeze(0)

        # convert NHWC → NCHW
        if img.shape[-1] in (3, 1):
            img = img.permute(0, 3, 1, 2)

        # remove invalid zero dimensions
        img = img[..., :max(1, img.shape[-2]), :max(1, img.shape[-1])]

        # ensure dimensions are multiples of 8
        h, w = img.shape[-2:]
        h8, w8 = (max(8, (h // 8) * 8), max(8, (w // 8) * 8))
        if (h, w) != (h8, w8):
            img = F.interpolate(img, size=(h8, w8), mode="bilinear", align_corners=False)

        # Convert NCHW → NHWC for encode()
        img = img.permute(0, 2, 3, 1)

        # Encode safely
        try:
            latent_m2_samples = vae2.encode(img)
        except Exception as e:
            raise RuntimeError(f"Cross-switch encode failed: {e}\nShape passed: {tuple(img.shape)}")
    else:
        latent_m2_samples = out1["samples"]
        
    # --- SECOND SAMPLING ---
    steps_after = max(total_steps - switch_point, 0)
    if steps_after > 0:
        out2 = _call_ksampler(m2, {"samples": latent_m2_samples}, steps_after, sampler_after, scheduler_after,
                             cfg_after, pos2, neg2, seed + 1, denoise_after)
    return out2

CrossStepSwitchKSampler = _make_node_class(
    "CrossStepSwitchKSampler",
    lambda: {
        "model1": ("MODEL",),
        "positive1": ("CONDITIONING",),
        "negative1": ("CONDITIONING",),
        "vae1": ("VAE",),
        "model2": ("MODEL",),
        "positive2": ("CONDITIONING",),
        "negative2": ("CONDITIONING",),
        "vae2": ("VAE",),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        "total_steps": ("INT", {"default": 20, "min": 2}),
        "switch_point": ("INT", {"default": 10, "min": 1}),
        "sampler_before": (tuple(cs.KSampler.SAMPLERS),),
        "sampler_after":  (tuple(cs.KSampler.SAMPLERS),),
        "scheduler_before": (tuple(cs.KSampler.SCHEDULERS),),
        "scheduler_after":  (tuple(cs.KSampler.SCHEDULERS),),
        "cfg_before": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
        "cfg_after":  ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
        "denoise_before": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "denoise_after":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "latent_image": ("LATENT",),
    },
    _cross_step_switch_handler
)
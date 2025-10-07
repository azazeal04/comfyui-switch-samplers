import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.samplers as cs
from .helpers import _call_ksampler, _make_node_class

def _cross_multistep_handler(model, latent, kwargs):
    device = mm.get_torch_device()

    # --- Stage setup ---
    steps = [kwargs.get("steps_stage1", 10),
             kwargs.get("steps_stage2", 10),
             kwargs.get("steps_stage3", 10)]

    samplers = [kwargs.get("sampler_stage1", cs.KSampler.SAMPLERS[0]),
                kwargs.get("sampler_stage2", cs.KSampler.SAMPLERS[0]),
                kwargs.get("sampler_stage3", cs.KSampler.SAMPLERS[0])]

    schedulers = [kwargs.get("scheduler_stage1", cs.KSampler.SCHEDULERS[0]),
                  kwargs.get("scheduler_stage2", cs.KSampler.SCHEDULERS[0]),
                  kwargs.get("scheduler_stage3", cs.KSampler.SCHEDULERS[0])]

    cfgs = [kwargs.get("cfg_stage1", 7.5),
            kwargs.get("cfg_stage2", 7.5),
            kwargs.get("cfg_stage3", 7.5)]

    denoises = [kwargs.get("denoise_stage1", 1.0),
                kwargs.get("denoise_stage2", 1.0),
                kwargs.get("denoise_stage3", 1.0)]

    seed = kwargs.get("seed", 0)

    # --- Models / VAEs / Conditioning ---
    m1 = kwargs.get("model1")
    m2 = kwargs.get("model2", m1)
    m3 = kwargs.get("model3", m1)

    vae1 = kwargs.get("vae1")
    vae2 = kwargs.get("vae2", vae1)
    vae3 = kwargs.get("vae3", vae2)

    pos1 = kwargs.get("positive1", kwargs.get("positive"))
    neg1 = kwargs.get("negative1", kwargs.get("negative"))
    pos2 = kwargs.get("positive2", pos1)
    neg2 = kwargs.get("negative2", neg1)
    pos3 = kwargs.get("positive3", pos1)
    neg3 = kwargs.get("negative3", neg1)

    # --- Stage 1 ---
    out1 = _call_ksampler(m1, latent, steps[0], samplers[0], schedulers[0],
                          cfgs[0], pos1, neg1, seed, denoises[0])

    # --- Between Stage 1 → 2 ---
    if vae1 is not None and vae2 is not None and vae1 != vae2:
        img = vae1.decode(out1["samples"])
        if img is None:
            raise RuntimeError("VAE1 decode returned None — cannot continue multi-step.")

        img = img if torch.is_tensor(img) else torch.tensor(img)
        if img.dim() == 5:
            b, f, h, w, c = img.shape
            img = img.view(b * f, h, w, c)
        elif img.dim() == 3:
            img = img.unsqueeze(0)

        if img.shape[-1] in (3, 1):
            img = img.permute(0, 3, 1, 2)

        img = img[..., :max(1, img.shape[-2]), :max(1, img.shape[-1])]
        h, w = img.shape[-2:]
        h8, w8 = (max(8, (h // 8) * 8), max(8, (w // 8) * 8))
        if (h, w) != (h8, w8):
            img = F.interpolate(img, size=(h8, w8), mode="bilinear", align_corners=False)

        img = img.permute(0, 2, 3, 1)
        try:
            latent2 = vae2.encode(img)
        except Exception as e:
            raise RuntimeError(f"Stage 1→2 encode failed: {e}\nShape: {tuple(img.shape)}")
    else:
        latent2 = out1["samples"]

    # --- Stage 2 ---
    out2 = _call_ksampler(m2, {"samples": latent2}, steps[1], samplers[1], schedulers[1],
                          cfgs[1], pos2, neg2, seed + 1, denoises[1])

    # --- Between Stage 2 → 3 ---
    if vae2 is not None and vae3 is not None and vae2 != vae3:
        img = vae2.decode(out2["samples"])
        if img is None:
            raise RuntimeError("VAE2 decode returned None — cannot continue multi-step.")

        img = img if torch.is_tensor(img) else torch.tensor(img)
        if img.dim() == 5:
            b, f, h, w, c = img.shape
            img = img.view(b * f, h, w, c)
        elif img.dim() == 3:
            img = img.unsqueeze(0)

        if img.shape[-1] in (3, 1):
            img = img.permute(0, 3, 1, 2)

        img = img[..., :max(1, img.shape[-2]), :max(1, img.shape[-1])]
        h, w = img.shape[-2:]
        h8, w8 = (max(8, (h // 8) * 8), max(8, (w // 8) * 8))
        if (h, w) != (h8, w8):
            img = F.interpolate(img, size=(h8, w8), mode="bilinear", align_corners=False)

        img = img.permute(0, 2, 3, 1)
        try:
            latent3 = vae3.encode(img)
        except Exception as e:
            raise RuntimeError(f"Stage 2→3 encode failed: {e}\nShape: {tuple(img.shape)}")
    else:
        latent3 = out2["samples"]

    # --- Stage 3 ---
    out3 = _call_ksampler(m3, {"samples": latent3}, steps[2], samplers[2], schedulers[2],
                          cfgs[2], pos3, neg3, seed + 2, denoises[2])

    return out3

CrossMultiStepKSampler = _make_node_class(
    "CrossMultiStepKSampler",
    lambda: {
        "model1": ("MODEL",),
        "positive1": ("CONDITIONING",),
        "negative1": ("CONDITIONING",),
        "vae1": ("VAE",),
        "model2": ("MODEL",),
        "positive2": ("CONDITIONING",),
        "negative2": ("CONDITIONING",),
        "vae2": ("VAE",),
        "model3": ("MODEL",),
        "positive3": ("CONDITIONING",),
        "negative3": ("CONDITIONING",),
        "vae3": ("VAE",),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        "steps_stage1": ("INT", {"default": 10, "min": 1}),
        "steps_stage2": ("INT", {"default": 10, "min": 1}),
        "steps_stage3": ("INT", {"default": 10, "min": 1}),
        "sampler_stage1": (tuple(cs.KSampler.SAMPLERS),),
        "sampler_stage2": (tuple(cs.KSampler.SAMPLERS),),
        "sampler_stage3": (tuple(cs.KSampler.SAMPLERS),),
        "scheduler_stage1": (tuple(cs.KSampler.SCHEDULERS),),
        "scheduler_stage2": (tuple(cs.KSampler.SCHEDULERS),),
        "scheduler_stage3": (tuple(cs.KSampler.SCHEDULERS),),
        "cfg_stage1": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
        "cfg_stage2": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
        "cfg_stage3": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
        "denoise_stage1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "denoise_stage2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "denoise_stage3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "latent_image": ("LATENT",),
    },
    _cross_multistep_handler
)
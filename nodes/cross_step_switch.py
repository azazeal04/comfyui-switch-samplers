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

    out = _call_ksampler(m1, latent, switch_point, sampler_before, scheduler_before,
                         cfg_before, pos1, neg1, seed, denoise_before)

    if m2 is not m1 and vae1 is not None and vae2 is not None:
        latent_samples = out["samples"] if isinstance(out, dict) else out
        img = vae1.decode(latent_samples)
        latent_m2_samples = vae2.encode(img)
        out = {"samples": latent_m2_samples}
    else:
        if not (isinstance(out, dict) and "samples" in out):
            out = {"samples": out}

    steps_after = max(total_steps - switch_point, 0)
    if steps_after > 0:
        out = _call_ksampler(m2, out, steps_after, sampler_after, scheduler_after,
                             cfg_after, pos2, neg2, seed + 1, denoise_after)
    return out

CrossStepSwitchKSampler = _make_node_class(
    "CrossStepSwitchKSampler",
    lambda: {
        "model1": ("MODEL",),
        "model2": ("MODEL",),
        "vae1": ("VAE",),
        "vae2": ("VAE",),
        "switch_point": ("INT", {"default": 10, "min": 1}),
        "total_steps": ("INT", {"default": 20, "min": 2}),
        "sampler_before": (tuple(cs.KSampler.SAMPLERS),),
        "sampler_after":  (tuple(cs.KSampler.SAMPLERS),),
        "scheduler_before": (tuple(cs.KSampler.SCHEDULERS),),
        "scheduler_after":  (tuple(cs.KSampler.SCHEDULERS),),
        "cfg_before": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
        "cfg_after":  ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
        "denoise_before": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "denoise_after":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        "positive1": ("CONDITIONING",),
        "negative1": ("CONDITIONING",),
        "positive2": ("CONDITIONING",),
        "negative2": ("CONDITIONING",),
        "latent_image": ("LATENT",),
    },
    _cross_step_switch_handler
)
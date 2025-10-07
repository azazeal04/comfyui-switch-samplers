from .helpers import _call_ksampler, _make_node_class
import comfy.samplers as cs

def _step_switch_handler(model, latent, kwargs):
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
    m2 = kwargs.get("model2", m1)  # default to m1 if not provided

    # Stage 1
    out = _call_ksampler(m1, latent, switch_point, sampler_before, scheduler_before,
                         cfg_before, kwargs.get("positive"), kwargs.get("negative"),
                         seed, denoise_before)

    # Stage 2
    steps_after = max(total_steps - switch_point, 0)
    if steps_after > 0:
        out = _call_ksampler(m2, out, steps_after, sampler_after, scheduler_after,
                             cfg_after, kwargs.get("positive"), kwargs.get("negative"),
                             seed + 1, denoise_after)

    return out



StepSwitchKSampler = _make_node_class("StepSwitchKSampler", lambda: {
    "model1": ("MODEL",),
    "model2": ("MODEL",),
    "positive": ("CONDITIONING",),
    "negative": ("CONDITIONING",),
    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
    "total_steps": ("INT", {"default": 20, "min": 2}),
    "switch_point": ("INT", {"default": 10, "min": 1}),
    "sampler_before": (tuple(cs.KSampler.SAMPLERS),),
    "sampler_after":  (tuple(cs.KSampler.SAMPLERS),),
    "scheduler_before": (tuple(cs.KSampler.SCHEDULERS),),
    "scheduler_after":  (tuple(cs.KSampler.SCHEDULERS),),
    "cfg_before": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
    "cfg_after":  ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
    "denoise_before": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
    "denoise_after":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
    "latent_image": ("LATENT",),

}, _step_switch_handler)

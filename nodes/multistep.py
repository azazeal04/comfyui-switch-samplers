from .helpers import _call_ksampler, _make_node_class, _schema_for_basic
import comfy.samplers as cs

def _multistep_handler(model, latent, kwargs):
    steps1 = kwargs.get("steps_stage1", 10)
    steps2 = kwargs.get("steps_stage2", 10)
    steps3 = kwargs.get("steps_stage3", 10)

    s1 = kwargs.get("sampler_stage1", cs.KSampler.SAMPLERS[0])
    s2 = kwargs.get("sampler_stage2", cs.KSampler.SAMPLERS[0])
    s3 = kwargs.get("sampler_stage3", cs.KSampler.SAMPLERS[0])

    sch1 = kwargs.get("scheduler_stage1", cs.KSampler.SCHEDULERS[0])
    sch2 = kwargs.get("scheduler_stage2", cs.KSampler.SCHEDULERS[0])
    sch3 = kwargs.get("scheduler_stage3", cs.KSampler.SCHEDULERS[0])

    cfg1 = kwargs.get("cfg_stage1", 7.50)
    cfg2 = kwargs.get("cfg_stage2", 7.50)
    cfg3 = kwargs.get("cfg_stage3", 7.50)

    d1 = kwargs.get("denoise_stage1", 1.00)
    d2 = kwargs.get("denoise_stage2", 1.00)
    d3 = kwargs.get("denoise_stage3", 1.00)

    seed = kwargs.get("seed", 0)

    # Models: fallback to earlier stage if missing
    m1 = kwargs.get("model1")
    m2 = kwargs.get("model2", m1)
    m3 = kwargs.get("model3", m2)

    out = latent
    if steps1 > 0:
        out = _call_ksampler(m1, out, steps1, s1, sch1, cfg1,
                             kwargs.get("positive"), kwargs.get("negative"),
                             seed, d1)
    if steps2 > 0:
        out = _call_ksampler(m2, out, steps2, s2, sch2, cfg2,
                             kwargs.get("positive"), kwargs.get("negative"),
                             seed + 1, d2)
    if steps3 > 0:
        out = _call_ksampler(m3, out, steps3, s3, sch3, cfg3,
                             kwargs.get("positive"), kwargs.get("negative"),
                             seed + 2, d3)

    return out




MultiStepKSampler = _make_node_class("MultiStepKSampler", lambda: {
    "model1": ("MODEL",),
    "model2": ("MODEL",),
    "model3": ("MODEL",),

    "steps_stage1": ("INT", {"default": 10, "min": 1}),
    "steps_stage2": ("INT", {"default": 10, "min": 1}),
    "steps_stage3": ("INT", {"default": 10, "min": 1}),
    
    "sampler_stage1": (tuple(cs.KSampler.SAMPLERS),),
    "sampler_stage2": (tuple(cs.KSampler.SAMPLERS),),
    "sampler_stage3": (tuple(cs.KSampler.SAMPLERS),),
    "scheduler_stage1": (tuple(cs.KSampler.SCHEDULERS),),
    "scheduler_stage2": (tuple(cs.KSampler.SCHEDULERS),),
    "scheduler_stage3": (tuple(cs.KSampler.SCHEDULERS),),

    "cfg_stage1": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
    "cfg_stage2": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
    "cfg_stage3": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),

    "denoise_stage1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
    "denoise_stage2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
    "denoise_stage3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
    "positive": ("CONDITIONING",),
    "negative": ("CONDITIONING",),
    "latent_image": ("LATENT",),
}, _multistep_handler)

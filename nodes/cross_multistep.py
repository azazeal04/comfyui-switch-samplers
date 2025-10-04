import comfy.samplers as cs
from .helpers import _call_ksampler, _make_node_class

def _cross_multistep_handler(model, latent, kwargs):
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

    models = [m1, m2, m3]
    vaes = [vae1, vae2, vae3]
    poss = [pos1, pos2, pos3]
    negs = [neg1, neg2, neg3]

    out = latent
    for i in range(3):
        if steps[i] <= 0:
            continue
        out = _call_ksampler(models[i], out, steps[i],
                             samplers[i], schedulers[i], cfgs[i],
                             poss[i], negs[i],
                             seed + i, denoises[i])

        if i < 2 and models[i+1] is not models[i] and vaes[i] and vaes[i+1]:
            latent_samples = out["samples"] if isinstance(out, dict) else out
            img = vaes[i].decode(latent_samples)
            new_latent_samples = vaes[i+1].encode(img)
            out = {"samples": new_latent_samples}
        else:
            if not (isinstance(out, dict) and "samples" in out):
                out = {"samples": out}
    return out

CrossMultiStepKSampler = _make_node_class(
    "CrossMultiStepKSampler",
    lambda: {
        "model1": ("MODEL",),
        "model2": ("MODEL",),
        "model3": ("MODEL",),
        "vae1": ("VAE",),
        "vae2": ("VAE",),
        "vae3": ("VAE",),
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
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        "positive1": ("CONDITIONING",),
        "negative1": ("CONDITIONING",),
        "positive2": ("CONDITIONING",),
        "negative2": ("CONDITIONING",),
        "positive3": ("CONDITIONING",),
        "negative3": ("CONDITIONING",),
        "latent_image": ("LATENT",),
    },
    _cross_multistep_handler
)
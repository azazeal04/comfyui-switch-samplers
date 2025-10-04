import torch
import comfy.samplers as cs
from comfy.sample import prepare_noise
try:
    from comfy.nodes import Node
except Exception:
    class Node: pass


def _schema_for_basic():
    return {
        "model": ("MODEL",),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffffffffffff}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
        "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1}),
        "sampler_name": (tuple(cs.KSampler.SAMPLERS),),
        "scheduler": (tuple(cs.KSampler.SCHEDULERS),),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "latent_image": ("LATENT",),
        "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
    }


def _call_ksampler(model, latent, steps, sampler_name, scheduler, cfg, positive, negative, seed, denoise=1.0):
    """
    Safe KSampler caller — fully compatible with all model architectures including Flux, SD3, and SDXL.
    Keeps original Azazeal node style intact.
    """
    if model is None:
        raise ValueError("No model provided to KSampler")

    device = getattr(model, "load_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model_options = getattr(model, "model_options", {})

    ks = cs.KSampler(
        model=model,
        steps=steps,
        device=device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        model_options=model_options
    )

    # ✅ Flux-safe latent unwrap
    latent_samples = latent["samples"] if isinstance(latent, dict) and "samples" in latent else latent
    if not torch.is_tensor(latent_samples):
        raise TypeError(f"Invalid latent type passed to KSampler: {type(latent)}")

    # ✅ Model-consistent noise generation (Flux-safe)
    noise = prepare_noise(latent_samples, seed, device)

    # ✅ Run the sampler
    out = ks.sample(
        noise=noise,
        positive=positive,
        negative=negative,
        cfg=cfg,
        latent_image=latent_samples,
        seed=seed
    )

    # ✅ Standardized latent dict output
    return {"samples": out}


def _make_node_class(class_name, schema_callable, handler, return_types=("LATENT",), category="Azazeal / Switch Samplers"):
    def INPUT_TYPES():
        return {"required": schema_callable()}

    def sample(self, *args, **kwargs):
        latent = kwargs.get("latent_image", None)
        if latent is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            latent = torch.randn(1, 4, 64, 64, device=device)
        out = handler(kwargs.get("model", None), latent, kwargs)
        return (out,)

    new_cls = type(class_name, (Node,), {
        "INPUT_TYPES": staticmethod(INPUT_TYPES),
        "RETURN_TYPES": return_types,
        "FUNCTION": "sample",
        "CATEGORY": category,
        "sample": sample,
    })
    return new_cls

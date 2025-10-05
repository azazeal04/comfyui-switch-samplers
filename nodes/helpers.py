import torch
import comfy.samplers as cs
import comfy.sample as csample
import comfy.model_management as mm

try:
    from comfy.nodes import Node
except Exception:
    class Node:
        pass


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


def _infer_expected_latent_channels(model):
    """
    Try several common locations to infer how many latent channels `model` expects.
    Returns int or None if unknown.
    """
    if model is None:
        return None

    checks = [
        lambda m: getattr(m, "latent_channels", None),
        lambda m: getattr(m, "ndim_latent", None),
        lambda m: getattr(getattr(m, "first_stage_model", None), "latent_channels", None),
        lambda m: getattr(getattr(m, "first_stage", None), "latent_channels", None),
        lambda m: getattr(getattr(m, "vae", None), "latent_channels", None),
        lambda m: getattr(getattr(m, "model_config", None), "latent_channels", None),
        lambda m: getattr(getattr(m, "model_config", None), "z_channels", None),
        lambda m: getattr(getattr(m, "diffusion_model", None), "in_channels", None),
        lambda m: getattr(getattr(m, "diffusion_model", None), "latent_channels", None),
    ]

    for fn in checks:
        try:
            v = fn(model)
            if isinstance(v, int) and v > 0:
                return int(v)
        except Exception:
            continue
    return None


def _format_latent_summary(latent_tensor):
    if not torch.is_tensor(latent_tensor):
        return f"type={type(latent_tensor)}"
    return f"shape={tuple(latent_tensor.shape)}, dtype={latent_tensor.dtype}, device={latent_tensor.device}"


def _fix_empty_latent_channels_fallback(model, latent_tensor):
    """
    Emulate comfy.sample.fix_empty_latent_channels if missing.
    Ensures latent has correct channel count and proper dtype/device.
    """
    expected = _infer_expected_latent_channels(model)
    if expected is None:
        return latent_tensor

    if not torch.is_tensor(latent_tensor):
        return latent_tensor

    latent = latent_tensor
    if latent.ndim != 4:
        # assume BCHW, add missing dims if needed
        while latent.ndim < 4:
            latent = latent.unsqueeze(0)

    b, c, h, w = latent.shape
    if c != expected:
        new_latent = torch.zeros((b, expected, h, w), dtype=latent.dtype, device=latent.device)
        new_latent[:, :min(c, expected), :, :] = latent[:, :min(c, expected), :, :]
        latent = new_latent

    return latent


def _call_ksampler(model, latent, steps, sampler_name, scheduler, cfg, positive, negative, seed, denoise=1.0):
    """
    Robust KSampler caller:
      - unwraps latent dicts,
      - fixes empty-latent channel mismatch via comfy.sample.fix_empty_latent_channels or fallback,
      - validates latent channels,
      - prepares noise correctly (supports batch_index),
      - uses comfy.sample.sample for full ComfyUI compatibility.
    """
    if model is None:
        raise ValueError("No model provided to KSampler")

    # --- DEVICE ---
    try:
        device_attr = getattr(model, "load_device", None)
        if callable(device_attr):
            device = device_attr()
        else:
            device = device_attr
        if device is None:
            device = mm.get_torch_device()
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- UNWRAP LATENT ---
    latent_samples = latent["samples"] if isinstance(latent, dict) and "samples" in latent else latent
    if not torch.is_tensor(latent_samples):
        raise TypeError(f"Invalid latent type: {type(latent)}. Expected torch.Tensor or dict with 'samples'.")

    # --- FIX EMPTY LATENT CHANNELS ---
    try:
        if hasattr(csample, "fix_empty_latent_channels"):
            latent_samples = csample.fix_empty_latent_channels(model, latent_samples)
        else:
            latent_samples = _fix_empty_latent_channels_fallback(model, latent_samples)
    except Exception:
        latent_samples = _fix_empty_latent_channels_fallback(model, latent_samples)

    # Move to device
    latent_samples = latent_samples.to(device).float()

    # --- VALIDATE CHANNEL COUNT ---
    expected_channels = _infer_expected_latent_channels(model)
    actual_channels = latent_samples.shape[1] if latent_samples.ndim >= 2 else None
    if expected_channels and actual_channels and expected_channels != actual_channels:
        raise RuntimeError(
            f"Latent channel mismatch: model expects {expected_channels} channels, "
            f"got {actual_channels} ({_format_latent_summary(latent_samples)})."
        )

    # --- NOISE / MASK HANDLING ---
    batch_inds = latent.get("batch_index", None) if isinstance(latent, dict) else None
    noise_mask = latent.get("noise_mask", None) if isinstance(latent, dict) else None

    try:
        noise = csample.prepare_noise(latent_samples, seed, batch_inds)
    except TypeError:
        noise = csample.prepare_noise(latent_samples, seed)

    # --- SAMPLING CALL ---
    samples = csample.sample(
        model=model,
        noise=noise,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent_samples,
        denoise=denoise,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        noise_mask=noise_mask,
        callback=None,
        disable_pbar=False,
        seed=seed
    )

    return {"samples": samples}


def _make_node_class(class_name, schema_callable, handler, return_types=("LATENT",), category="Azazeal / Switch Samplers"):
    def INPUT_TYPES():
        return {"required": schema_callable()}

    def sample(self, *args, **kwargs):
        latent = kwargs.get("latent_image", None)
        if latent is None:
            raise RuntimeError("No latent input provided. Connect an Empty Latent node.")
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

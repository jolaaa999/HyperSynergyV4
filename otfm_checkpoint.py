"""Explicit, strict legacy/V4 model loading and training-state persistence."""
import random
import numpy as np
import torch
from prior_config import PriorConfig

FORMAT = "hyperV4_otfm_v1"


def checkpoint_payload(model, step, optimizer=None, optimizer_lr=None, scheduler=None):
    payload = {
        "step": step, "state_dict": model.state_dict(),
        "prior_format": FORMAT, "prior_config": model.prior_config.to_dict(),
    }
    for name, component in (("optimizer", optimizer), ("optimizer_lr", optimizer_lr),
                            ("scheduler", scheduler)):
        if component is not None:
            payload[name] = component.state_dict()
    payload["rng"] = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return payload


def load_checkpoint(model, payload, optimizer=None, optimizer_lr=None,
                    scheduler=None, restore_rng=False):
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("Expected a checkpoint containing state_dict")
    state = payload["state_dict"]
    if "prior_format" in payload or "prior_config" in payload:
        if payload.get("prior_format") != FORMAT or "prior_config" not in payload:
            raise ValueError("Unrecognized or incomplete V4 prior metadata")
        config = PriorConfig.from_dict(payload["prior_config"])
    else:
        if any(key.startswith("otfm.") for key in state):
            raise ValueError("OT-FM weights require explicit V4 prior metadata")
        config = PriorConfig()
    components = (("optimizer", optimizer), ("optimizer_lr", optimizer_lr),
                  ("scheduler", scheduler))
    if any(component is not None for _, component in components):
        if config != model.prior_config:
            raise ValueError("Construct the checkpoint prior before constructing optimizers")
        for name, component in components:
            if component is not None and name not in payload:
                raise ValueError("Checkpoint lacks training state: " + name)
    if restore_rng and "rng" not in payload:
        raise ValueError("Checkpoint lacks RNG state")
    model.configure_prior(config)
    model.load_state_dict(state, strict=True)
    for name, component in components:
        if component is not None:
            component.load_state_dict(payload[name])
    if restore_rng:
        rng = payload["rng"]
        if len(rng["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
    return payload.get("step", 0)

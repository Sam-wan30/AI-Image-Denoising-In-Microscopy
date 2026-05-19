"""Lightweight model helpers (no matplotlib / heavy deps)."""


def detect_model_type(state_dict_keys: list[str]) -> str:
    """Infer architecture from checkpoint keys."""
    if any(k.startswith("unet.") for k in state_dict_keys):
        return "residual"
    if any("residual_blocks" in k for k in state_dict_keys):
        return "enhanced"
    return "standard"

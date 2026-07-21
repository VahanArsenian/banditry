"""Publication-quality display names for agents.

Display strings are derived from *structural* attributes
(`frequentist`, `surrogate_name`, ...) rather than persisted as a flat
string, so a cosmetic rename here re-renders everywhere without
touching any stored artefacts.
"""

from __future__ import annotations

from typing import Optional


def _ofugp_name(params: dict) -> str:
    frequentist = bool(params.get("frequentist"))
    flavour = "Frequentist" if frequentist else "Bayesian"
    surrogate = str(params.get("surrogate_name", "?")).upper()
    base = f"GP-UCB ({flavour}, {surrogate})"
    if frequentist:
        rkhs = params.get("rkhs_norm")
        if rkhs is not None:
            base += rf" [$b_i={float(rkhs):.3g}$]"
    return base


def _ts_name(params: dict) -> str:
    sampler_raw = str(params.get("sampler_name", "?"))
    sampler = sampler_raw.removesuffix("Sampler") or sampler_raw
    nll_raw = params.get("nll_name") or "Gaussian"
    nll = str(nll_raw).removesuffix("NLL") or nll_raw
    base = f"Thompson Sampling ({sampler}, {nll} NLL)"
    latent = params.get("latent_dimension")
    if latent is not None:
        base += rf" [$d={float(latent):.3g}$]"
    return base


_AGENT_DISPATCH = {
    "OFUGPAgent": _ofugp_name,
    "TSAgent": _ts_name,
}


def agent_display_name(agent_type: str, params: Optional[dict] = None) -> str:
    """Publication name from `(class_name, structural_params)`. Unknown
    types fall back to the class name verbatim — a new agent renders
    something readable before its case is added here."""
    fn = _AGENT_DISPATCH.get(agent_type)
    return fn(params or {}) if fn is not None else agent_type

#!/usr/bin/env python3
"""Fail-loud Qwen3.5 model-class guard for text-only training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM


TEXT_DERIVATIVE_MARKERS = (
    "qwen3_5_text",
    "qwen3.5_text",
    "qwen3.5-text",
    "qwen35_text",
)
RISK_MARKERS = ("conditionalgeneration", "vision", "visual", "multimodal", "multi_modal")
MTP_ATTRS = ("mtp_config", "num_nextn_predict_layers", "num_nextn_predict_layer", "mtp_depth")
PARITY_PROOF_ENV = "QWEN35_TEXT_PARITY_PROOF"
PARITY_PROOF_FILES = ("qwen3_5_text_parity.json", "text_logit_parity.json")
DEFAULT_PARITY_TOLERANCE = 1e-4


class ModelClassGuardError(RuntimeError):
    pass


def _config_architectures(config: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in (getattr(config, "architectures", None) or ()))


def _risk_reasons(config: Any) -> list[str]:
    archs = _config_architectures(config)
    tokens = [*archs, str(getattr(config, "model_type", ""))]
    blob = " ".join(tokens).lower()
    reasons: list[str] = []
    for marker in RISK_MARKERS:
        if marker in blob:
            reasons.append(f"config marker '{marker}' in architectures/model_type")
    if "mtp" in blob:
        reasons.append("MTP marker in architectures/model_type")
    for attr in ("vision_config", "visual_config", "vision_tower", "mm_projector", "multi_modal_projector"):
        if getattr(config, attr, None) is not None:
            reasons.append(f"config has {attr}")
    for attr in MTP_ATTRS:
        value = getattr(config, attr, None)
        if value not in (None, 0, False):
            reasons.append(f"config has {attr}={value!r}")
    return reasons


def _proof_paths(model_path: str | os.PathLike[str]) -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get(PARITY_PROOF_ENV)
    if env_path:
        paths.append(Path(env_path))
    local_path = Path(model_path)
    if local_path.is_dir():
        paths.extend(local_path / name for name in PARITY_PROOF_FILES)
    return paths


def _load_parity_proof(model_path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    searched = _proof_paths(model_path)
    for path in searched:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            proof = json.load(handle)
        if not isinstance(proof, dict):
            raise ModelClassGuardError(f"parity proof {path} must be a JSON object")
        return path, proof
    searched_text = ", ".join(str(path) for path in searched) or f"${PARITY_PROOF_ENV} or local proof file"
    raise ModelClassGuardError(
        "refusing text-only Qwen3.5 training without text-logit parity proof; "
        f"expected {searched_text}"
    )


def _contains_text_marker(*values: Any) -> bool:
    blob = " ".join(str(value) for value in values if value is not None).lower()
    return any(marker in blob for marker in TEXT_DERIVATIVE_MARKERS)


def _numeric_proof_value(proof: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = proof.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ModelClassGuardError(f"parity proof field {key!r} must be numeric, got {value!r}") from exc
    return None


def _assert_documented_text_derivative(
    model_path: str | os.PathLike[str],
    config: Any,
) -> tuple[Path, dict[str, Any]]:
    proof_path, proof = _load_parity_proof(model_path)
    archs = _config_architectures(config)
    if not _contains_text_marker(
        model_path,
        getattr(config, "model_type", None),
        archs,
        proof.get("model_class"),
        proof.get("derivative_type"),
        proof.get("artifact_type"),
    ):
        raise ModelClassGuardError(
            "refusing AutoModelForCausalLM load: artifact is not documented as a "
            f"qwen3_5_text derivative in model path/config/parity proof ({proof_path})"
        )

    passed = proof.get("passed") is True or proof.get("text_logit_parity") is True
    if not passed:
        raise ModelClassGuardError(f"parity proof {proof_path} does not declare passed=true")
    prompt_hash = proof.get("prompt_sha256") or proof.get("frozen_prompt_sha256")
    if not isinstance(prompt_hash, str) or not prompt_hash:
        raise ModelClassGuardError(f"parity proof {proof_path} missing frozen prompt hash")
    max_delta = _numeric_proof_value(
        proof,
        "max_abs_logit_delta",
        "max_abs_diff",
        "max_abs_error",
    )
    if max_delta is None:
        raise ModelClassGuardError(f"parity proof {proof_path} missing max_abs_logit_delta")
    tolerance = _numeric_proof_value(proof, "tolerance", "threshold") or DEFAULT_PARITY_TOLERANCE
    if max_delta > tolerance:
        raise ModelClassGuardError(
            f"parity proof {proof_path} failed tolerance: max_abs_logit_delta={max_delta} > {tolerance}"
        )
    return proof_path, proof


def load_qwen35_text_causal_lm_checked(
    model_path: str | os.PathLike[str],
    **from_pretrained_kwargs: Any,
) -> Any:
    """Load a Qwen3.5 text-only derivative only after config and parity checks."""
    trust_remote_code = bool(from_pretrained_kwargs.get("trust_remote_code", False))
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    archs = _config_architectures(config)
    risk_reasons = _risk_reasons(config)
    if risk_reasons:
        raise ModelClassGuardError(
            "refusing to coerce a non-text Qwen3.5 artifact through AutoModelForCausalLM; "
            f"architectures={archs or ('<missing>',)}, model_type={getattr(config, 'model_type', None)!r}, "
            f"reasons={risk_reasons}. Load the named ConditionalGeneration class and freeze vision/MTP, "
            "or provide a documented qwen3_5_text derivative with text-logit parity proof."
        )

    proof_path, proof = _assert_documented_text_derivative(model_path, config)
    print(
        "[model-class-guard] verified qwen3_5_text derivative "
        f"architectures={archs or ('<missing>',)} parity_proof={proof_path} "
        f"prompt_sha256={proof.get('prompt_sha256') or proof.get('frozen_prompt_sha256')}"
    )
    return AutoModelForCausalLM.from_pretrained(model_path, **from_pretrained_kwargs)

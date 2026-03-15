from __future__ import annotations

BACKEND_NAME = "triton"


def is_available() -> tuple[bool, str]:
    return False, "Triton Fused MoE kernel is reserved as a research extension point and has not been implemented yet."


def run(inputs):
    raise NotImplementedError("Fused MoE Triton kernel is not implemented yet.")

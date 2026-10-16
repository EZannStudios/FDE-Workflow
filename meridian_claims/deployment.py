"""Rollout controls: shadow -> assist -> canary auto-approve, with a kill switch.

  SHADOW  AI runs on live claims; results are logged and compared with what
          adjusters decided. Nothing a person sees changes.
  ASSIST  AI pre-fills every claim file; every claim still goes to a person.
  AUTO    Clean claims under the auto-approve limit skip the queue, but only
          for the canary slice. Everyone else stays on ASSIST.

Moving between stages is a config change, gated on the eval report.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import Enum

from .models import Route


class Mode(str, Enum):
    SHADOW = "shadow"
    ASSIST = "assist"
    AUTO = "auto"


@dataclass(frozen=True)
class RolloutConfig:
    mode: Mode = Mode.SHADOW
    canary_percent: int = 0  # share of claims eligible for AUTO, 0-100

    @classmethod
    def from_env(cls) -> "RolloutConfig":
        if os.environ.get("MERIDIAN_AI_KILL_SWITCH") == "1":
            # One env var returns the whole operation to the pre-AI process.
            return cls(mode=Mode.SHADOW, canary_percent=0)
        return cls(
            mode=Mode(os.environ.get("MERIDIAN_AI_MODE", Mode.SHADOW.value)),
            canary_percent=int(os.environ.get("MERIDIAN_CANARY_PERCENT", "0")),
        )


def in_canary(claim_id: str, percent: int) -> bool:
    """Stable bucketing: the same claim always lands in the same bucket, so a
    retry can't flip a claim between auto and manual."""
    bucket = int(hashlib.sha256(claim_id.encode()).hexdigest(), 16) % 100
    return bucket < percent


def effective_route(proposed: Route, claim_id: str, config: RolloutConfig) -> Route:
    """Downgrade the policy's route to what the current rollout stage allows."""
    if proposed is Route.AUTO_APPROVE:
        if config.mode is not Mode.AUTO or not in_canary(claim_id, config.canary_percent):
            return Route.ADJUSTER_REVIEW
    return proposed


def ai_is_visible(config: RolloutConfig) -> bool:
    """In shadow mode, adjusters work the claim exactly as before."""
    return config.mode is not Mode.SHADOW

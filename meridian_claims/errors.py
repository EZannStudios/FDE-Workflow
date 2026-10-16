"""Error taxonomy and the retry/circuit-breaker helpers used around AI calls.

The design rule: an AI failure never blocks or decides a claim. Every failure
mode below ends in one of two places: a retry, or the human review queue.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")


class AIStepError(Exception):
    """Base class: something about an AI step's output can't be trusted."""


class TransientModelError(AIStepError):
    """Timeout, rate limit, 5xx. Worth retrying."""


class MalformedOutputError(AIStepError):
    """Output wasn't valid JSON or didn't match the schema. Retried, then human."""


class CircuitOpenError(AIStepError):
    """Too many recent failures; stop calling the model and route to humans."""


class SystemOfRecordError(Exception):
    """An existing system (TMS, ERP, doc store) failed. Not an AI problem; never guess around it."""


def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    retry_on: tuple[type[Exception], ...] = (TransientModelError, MalformedOutputError),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, retrying listed errors with exponential backoff and jitter.

    Anything not in `retry_on` is raised immediately. In particular a
    SystemOfRecordError is not retried here: if the TMS returned bad data,
    asking again won't fix it, and a person needs to look.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1) * (1 + random.random() * 0.25)
            log.warning("attempt %d/%d failed (%s); retrying in %.2fs", attempt, attempts, exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")


class CircuitBreaker:
    """Opens after `threshold` consecutive failures, stays open for `cooldown_s`.

    While open, the pipeline skips AI entirely and sends every claim to the
    manual queue, which is exactly how Meridian worked before this project.
    That makes "the model is down" a capacity problem, not an outage.
    """

    def __init__(self, threshold: int = 5, cooldown_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    def call(self, fn: Callable[[], T]) -> T:
        if self._opened_at is not None:
            if self._clock() - self._opened_at < self.cooldown_s:
                raise CircuitOpenError("model circuit open; using manual path")
            self._opened_at = None  # half-open: let one call through
        try:
            result = fn()
        except AIStepError:
            self._failures += 1
            if self._failures >= self.threshold:
                self._opened_at = self._clock()
                log.error("circuit opened after %d consecutive AI failures", self._failures)
            raise
        self._failures = 0
        return result

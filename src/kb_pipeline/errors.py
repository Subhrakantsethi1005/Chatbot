"""Exception types used across the knowledge-base pipeline."""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every error raised by the pipeline."""


class ConfigError(PipelineError):
    """The configuration file is missing or has an invalid value."""


class TransientError(PipelineError):
    """A temporary failure (outage, locked file, unreachable source).

    Updates that fail with a transient error are retried on the configured
    retry schedule (15, 30 and 60 minutes by default).
    """


class LockHeldError(TransientError):
    """Another pipeline run currently holds the update lock."""


class StoreError(PipelineError):
    """A knowledge-base version is missing, corrupted or cannot be written."""


class ValidationError(PipelineError):
    """A source document is invalid and must be quarantined."""

    def __init__(self, reason: str, detail: str | list[str] = "") -> None:
        self.reason = reason
        self.details = [detail] if isinstance(detail, str) and detail else list(detail or [])
        message = reason if not self.details else f"{reason}: {'; '.join(self.details[:5])}"
        super().__init__(message)


class AccessDenied(PipelineError):
    """The caller is not authenticated or lacks the required permission."""


class AuthLockedOut(AccessDenied):
    """Too many failed authentication attempts from this client."""

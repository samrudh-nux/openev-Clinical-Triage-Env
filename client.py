from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import httpx

# Re-export models so `from client import TriageAction` works
try:
    from models import (
        TriageAction,
        TriageObservation,
        MedicationSafetyAction,
        MedicationSafetyObservation,
        SepsisManagementAction,
        SepsisManagementObservation,
    )
except ImportError:
    from .models import (  # type: ignore[no-redef]
        TriageAction,
        TriageObservation,
        MedicationSafetyAction,
        MedicationSafetyObservation,
        SepsisManagementAction,
        SepsisManagementObservation,
    )

__all__ = [
    "ClinicalTriageEnv",
    "TriageAction",
    "TriageObservation",
    "MedicationSafetyAction",
    "MedicationSafetyObservation",
    "SepsisManagementAction",
    "SepsisManagementObservation",
]

AnyAction = TriageAction | MedicationSafetyAction | SepsisManagementAction


class ClinicalTriageEnv:
    """
    Async HTTP client for the ClinicalTriageEnv OpenEnv environment.
    Supports context manager usage:
        async with ClinicalTriageEnv(base_url="...") as env:
            obs = await env.reset(task_id="triage_easy")
    """

    def __init__(
        self,
        base_url: str = "https://samrudh-nux-clinicaltriageenv.hf.space",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "ClinicalTriageEnv":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "ClinicalTriageEnv must be used as an async context manager: "
                "`async with ClinicalTriageEnv(...) as env:`"
            )
        return self._client

    async def health(self) -> dict[str, Any]:
        """Check server health."""
        resp = await self._get_client().get("/health")
        resp.raise_for_status()
        return resp.json()

    async def reset(
        self,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Reset the environment and start a new episode.
        Args:
            task_id: Specific task ID (e.g. "triage_easy", "sepsis_hard").
                     If None, a random task is selected.
            difficulty: Filter by difficulty ("easy", "medium", "hard").
            task_type: Filter by type ("triage", "medication_safety", "sepsis").
        Returns:
            Observation dict with patient/scenario data.
        """
        payload: dict[str, Any] = {}
        if task_id is not None:
            payload["task_id"] = task_id
        if difficulty is not None:
            payload["difficulty"] = difficulty
        if task_type is not None:
            payload["task_type"] = task_type

        resp = await self._get_client().post("/reset", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def step(self, action: AnyAction) -> dict[str, Any]:
        """
        Submit an action to the environment.
        Args:
            action: A TriageAction, MedicationSafetyAction, or SepsisManagementAction.
        Returns:
            dict with keys: observation, reward, done, info
        """
        resp = await self._get_client().post(
            "/step",
            json=action.model_dump(),
        )
        resp.raise_for_status()
        return resp.json()

    async def state(self) -> dict[str, Any]:
        """Get the current environment state."""
        resp = await self._get_client().get("/state")
        resp.raise_for_status()
        return resp.json()

    async def list_tasks(self) -> list[dict[str, Any]]:
        """List all available tasks."""
        resp = await self._get_client().get("/tasks")
        resp.raise_for_status()
        return resp.json()


    def sync(self) -> "_SyncClinicalTriageEnv":
        """Return a synchronous wrapper around this client."""
        return _SyncClinicalTriageEnv(self)


class _SyncClinicalTriageEnv:
    """Synchronous wrapper for ClinicalTriageEnv."""

    def __init__(self, async_env: ClinicalTriageEnv) -> None:
        self._env = async_env
        self._loop = asyncio.new_event_loop()

    def __enter__(self) -> "_SyncClinicalTriageEnv":
        self._loop.run_until_complete(self._env.__aenter__())
        return self

    def __exit__(self, *args: Any) -> None:
        self._loop.run_until_complete(self._env.__aexit__(*args))
        self._loop.close()

    def health(self) -> dict[str, Any]:
        return self._loop.run_until_complete(self._env.health())

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        return self._loop.run_until_complete(self._env.reset(**kwargs))

    def step(self, action: AnyAction) -> dict[str, Any]:
        return self._loop.run_until_complete(self._env.step(action))

    def state(self) -> dict[str, Any]:
        return self._loop.run_until_complete(self._env.state())

    def list_tasks(self) -> list[dict[str, Any]]:
        return self._loop.run_until_complete(self._env.list_tasks())

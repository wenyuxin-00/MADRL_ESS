from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from REMAKE.configs.cfg import Cfg
from REMAKE.controllers.madrl_safety import JointGridSafetyProjector, enforce_local_action_feasibility, map_actor_output_to_soc_feasible_action
from REMAKE.models.assembly import Actor, build_actors, to_torch_obs
from REMAKE.utils.torch_runtime import resolve_device


def _batched_obs(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0) for key, value in obs.items()}


class MADRLController:
    def __init__(self, cfg: Cfg, actors: list[Actor], *, name: str, projection: bool) -> None:
        self.cfg, self.actors, self.name, self.projection = cfg, actors, str(name), bool(projection)
        self.device = next(self.actors[0].parameters()).device
        self.projector = JointGridSafetyProjector(cfg).to(self.device) if self.projection else None

    def reset(self, env_state: dict) -> None:
        self.env_state = dict(env_state)

    def _raw_actor_action(self, obs_t: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([actor(obs_t) for actor in self.actors], dim=1)

    def _postprocess(self, obs_t: dict[str, torch.Tensor], raw_action: torch.Tensor) -> torch.Tensor:
        action = map_actor_output_to_soc_feasible_action(self.cfg, obs_t["safety_local"], raw_action)
        if self.projector is not None:
            action = self.projector.project_actions_from_safety_local(obs_t["safety_local"], action)
        return enforce_local_action_feasibility(self.cfg, obs_t["safety_local"], action)

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        obs_t = _batched_obs(obs, self.device)
        with torch.inference_mode():
            action = self._postprocess(obs_t, self._raw_actor_action(obs_t))[0]
        return action.detach().cpu().numpy().astype(np.float32)

    @classmethod
    def load(cls, cfg: Cfg, path: str | Path) -> "MADRLController":
        device = resolve_device(cfg.runtime.device)
        payload = torch.load(Path(path), map_location=device)
        actors = [actor.to(device) for actor in build_actors(cfg)]
        for actor, state in zip(actors, payload["actors"], strict=True):
            actor.load_state_dict(state); actor.eval()
        meta = dict(payload["meta"])
        return cls(cfg, actors, name=str(meta.get("controller", cfg.algo.name)), projection=bool(meta["projection"]))

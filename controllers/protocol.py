from __future__ import annotations

from pathlib import Path
from typing import Protocol

from configs.cfg import Cfg


class Controller(Protocol):
    name: str
    def reset(self, env_state: dict) -> None: ...


CONTROLLER_NAMES = ("MISOCP", "LOCAL_MPC", "ADMM_MPC", "MADRL_BASE", "MADRL_PENALTY", "MADRL_PROJECTION")


def build_controller(name: str, cfg: Cfg, model_path: Path | None = None) -> Controller:
    from controllers.mpc_admm import AdmmMpcController
    from controllers.mpc_global import MisocpController
    from controllers.mpc_local import LocalMPCController
    from controllers.madrl import MADRLController
    return {
        "LOCAL_MPC": lambda: LocalMPCController(cfg),
        "ADMM_MPC": lambda: AdmmMpcController(cfg),
        "MISOCP": lambda: MisocpController(cfg),
        "MADRL_BASE": lambda: MADRLController.load(cfg.with_algo(name), model_path),
        "MADRL_PENALTY": lambda: MADRLController.load(cfg.with_algo(name), model_path),
        "MADRL_PROJECTION": lambda: MADRLController.load(cfg.with_algo(name), model_path),
    }[name]()

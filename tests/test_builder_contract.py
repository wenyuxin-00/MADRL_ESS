import pytest

from configs.default_config import Config
from core.builder import resolve_algorithm_model_config


def test_maddpg_resolves_default_critic():
    args = Config()
    args.algorithm = "MADDPG"
    args.critic_type = None

    resolve_algorithm_model_config(args)

    assert args.critic_type == "maddpg_mlp"


def test_matd3_resolves_default_critic():
    args = Config()
    args.algorithm = "MATD3"
    args.critic_type = None

    resolve_algorithm_model_config(args)

    assert args.critic_type == "matd3_mlp"


def test_incompatible_algorithm_and_critic_fails_fast():
    args = Config()
    args.algorithm = "MATD3"
    args.critic_type = "maddpg_mlp"

    with pytest.raises(ValueError, match="incompatible"):
        resolve_algorithm_model_config(args)

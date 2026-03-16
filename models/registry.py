"""Registry and builders for actor/critic models."""

ACTOR_REGISTRY: dict = {}
CRITIC_REGISTRY: dict = {}


def register_actor(name: str):
    """Register an actor model under a string key."""

    def decorator(cls):
        ACTOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_critic(name: str):
    """Register a critic model under a string key."""

    def decorator(cls):
        CRITIC_REGISTRY[name] = cls
        return cls

    return decorator


def build_actor(args, agent_id):
    """Instantiate the actor selected by ``args.actor_type``."""
    actor_type = getattr(args, "actor_type", "mlp")
    if actor_type not in ACTOR_REGISTRY:
        raise ValueError(f"Unknown actor_type '{actor_type}', available: {list(ACTOR_REGISTRY)}")
    return ACTOR_REGISTRY[actor_type](args, agent_id)


def build_critic(args):
    """Instantiate the critic selected by ``args.critic_type``."""
    critic_type = getattr(args, "critic_type", None)
    if critic_type is None:
        raise ValueError("args.critic_type is required to build a critic.")
    if critic_type not in CRITIC_REGISTRY:
        raise ValueError(f"Unknown critic_type '{critic_type}', available: {list(CRITIC_REGISTRY)}")
    return CRITIC_REGISTRY[critic_type](args)


# Import concrete model modules after registry helpers are defined so they can
# register themselves on import.
import models.actors  # noqa: E402,F401
import models.critics  # noqa: E402,F401

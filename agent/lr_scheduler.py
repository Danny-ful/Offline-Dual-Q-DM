"""Learning-rate scheduler helpers shared by actor-based agents."""

import torch


def make_actor_lr_scheduler(optimizer, args, initial_lr):
    """Build an optional linear scheduler over actor optimizer updates.

    ``final_lr`` is an absolute learning rate, not a multiplier. Computing the
    multiplier from ``initial_lr`` keeps the schedule correct for arbitrary
    actor learning rates selected by Hydra or a W&B sweep.
    """
    scheduler_cfg = getattr(args, "actor_lr_scheduler", None)
    enabled = bool(getattr(scheduler_cfg, "enabled", False))

    # Backwards compatibility for existing commands using the old misspelled
    # top-level flag. New configurations should use actor_lr_scheduler.enabled.
    enabled = enabled or bool(getattr(args, "schedular", False))
    if not enabled:
        return None

    initial_lr = float(initial_lr)
    final_lr = float(getattr(scheduler_cfg, "final_lr", 3e-6))
    anneal_updates = int(getattr(scheduler_cfg, "anneal_updates", 300000))

    if initial_lr <= 0:
        raise ValueError("agent.actor_lr must be positive")
    if not 0 <= final_lr <= initial_lr:
        raise ValueError(
            "actor_lr_scheduler.final_lr must satisfy "
            "0 <= final_lr <= agent.actor_lr"
        )
    if anneal_updates <= 0:
        raise ValueError("actor_lr_scheduler.anneal_updates must be positive")

    final_ratio = final_lr / initial_lr

    def lr_multiplier(update_idx):
        if update_idx >= anneal_updates:
            return final_ratio
        progress = update_idx / anneal_updates
        return 1.0 + progress * (final_ratio - 1.0)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_multiplier,
    )


def step_actor_lr_scheduler(scheduler):
    """Advance a configured actor scheduler by one optimizer update."""
    if scheduler is not None:
        scheduler.step()

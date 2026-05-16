"""Minimal Behavior Cloning update for SAC-style continuous agents.

The core loop is intentionally kept as a free function so it can be attached to
an existing SAC agent via ``types.MethodType`` (mirroring how ``iq_update`` is
wired up in ``train_iq.py``). This keeps the training script almost unchanged
while isolating BC-specific logic here.
"""

import torch
import torch.nn.functional as F


def bc_update(self, expert_buffer, logger, step):
    """Single BC optimizer step using only expert data.

    The agent's actor/optimizer/scheduler are reused (no new modules created).
    Supports two loss flavors selected by ``args.method.loss``:
      - ``nll``: negative log-likelihood of expert actions under the SAC policy
        (SquashedNormal). Expert actions are clipped slightly inside (-1, 1)
        for numerical safety of the tanh-Normal log_prob.
      - ``mse``: mean-squared error between ``actor(obs).mean`` and expert
        actions (deterministic behavior cloning).
    """
    args = self.args
    obs, _next_obs, action, _reward, _done = expert_buffer.get_samples(
        self.batch_size, self.device)

    loss_type = getattr(args.method, "loss", "nll")
    dist = self.actor(obs)

    if loss_type == "nll":
        eps = 1e-6
        a = action.clamp(-1.0 + eps, 1.0 - eps)
        log_prob = dist.log_prob(a).sum(-1, keepdim=True)
        actor_loss = -log_prob.mean()
    elif loss_type == "mse":
        actor_loss = F.mse_loss(dist.mean, action)
    else:
        raise ValueError(f"Unknown BC loss: {loss_type}")

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()
    if self.she:
        self.scheduler.step()

    logger.log("train/bc_loss", actor_loss, step)

    losses = {
        "loss/bc": actor_loss.item(),
        "lr": self.actor_optimizer.param_groups[0]["lr"],
    }

    if loss_type == "nll":
        with torch.no_grad():
            losses["bc/action_mse"] = F.mse_loss(dist.mean, action).item()
    return losses

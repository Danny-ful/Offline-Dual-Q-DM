"""ReCOIL offline imitation learning objective.

Implements the three losses from the ReCOIL paper (Algorithm 1, Eqs. 11-13):
  * `recoil_q_loss`    : semi-gradient chi^2 critic update (Eq. 11)
  * `recoil_v_loss`    : implicit-maximum state-value update (Eq. 12)
  * `recoil_actor_loss`: advantage-weighted regression policy update (Eq. 13)

The orchestrator `recoil_update` mirrors the layout of `iq_update` in
`train_iq.py`, so it can be attached to a ReCOIL agent via
``types.MethodType`` just like the IQ-Learn / BC updates.
"""
import torch

from utils.utils import get_concat_samples, soft_update, average_dicts


_ACTION_LOGPROB_EPS = 1e-6


def _recoil_critic_only(agent):
    method_cfg = agent.args.method
    return bool(getattr(method_cfg, "recoil_critic_only", False))


def _split_batch(batch):
    obs, next_obs, action, _reward, done, is_expert = batch
    return obs, next_obs, action, done, is_expert


def _target_q(agent, obs, action):
    """Target-critic Q estimate with DoubleQ fallback handled."""
    try:
        q1, q2 = agent.critic_target(obs, action, both=True)
        return torch.min(q1, q2)
    except TypeError:
        return agent.critic_target(obs, action)


def _target_v_from_policy(agent, obs, num_samples=1):
    """Monte-Carlo baseline: E_{a~pi(.|s)}[Q_target(s,a)] without gradients."""
    num_samples = max(1, int(num_samples))
    v_hat = None
    for _ in range(num_samples):
        sampled_action, _log_prob, _ = agent.actor.sample(obs)
        q = _target_q(agent, obs, sampled_action)
        v_hat = q if v_hat is None else (v_hat + q)
    return v_hat / float(num_samples)


def _target_advantage(agent, obs, action):
    """A(s,a_data)=Q_target(s,a_data)-mean_{a'~pi}[Q_target(s,a')], fully detached."""
    num_samples = int(getattr(agent.args.method, "actor_baseline_samples", 4))
    with torch.no_grad():
        q_data = _target_q(agent, obs, action)
        v_hat = _target_v_from_policy(agent, obs, num_samples=num_samples)
    return q_data - v_hat


def recoil_q_loss(agent, current_Q, next_V, batch):
    """Eq. 11: beta*(E_{d^S, pi}[Q] - E_{d^E}[Q]) + q_coef * E_{mix}[(gamma V(s') - Q)^2].

    Args:
        agent: ReCOIL agent instance.
        current_Q: tensor [N, 1] from critic(obs, action) for the mixture batch.
        next_V: tensor [N, 1], already detached.
        batch: concatenated batch tuple from get_concat_samples.
    """
    args = agent.args
    gamma = agent.gamma
    obs, _next_obs, _action, done, is_expert = _split_batch(batch)

    # First term: sampled-policy Q on suboptimal states minus expert data Q.
    with torch.no_grad():
        policy_action, _log_prob, _ = agent.actor.sample(obs)
    Q_on_policy = agent.critic(obs, policy_action)

    policy_mask = (~is_expert).squeeze(-1)
    expert_mask = is_expert.squeeze(-1)

    if policy_mask.any():
        q_policy_term = Q_on_policy[policy_mask].mean()
    else:
        q_policy_term = current_Q.new_tensor(0.0)
    if expert_mask.any():
        q_expert_term = current_Q[expert_mask].mean()
    else:
        q_expert_term = current_Q.new_tensor(0.0)

    linear_term = args.method.beta * (q_policy_term - q_expert_term)

    # Semi-gradient Bellman residual over the full mixture.
    target = (1.0 - done) * gamma * next_V
    bellman = ((target - current_Q) ** 2).mean()
    loss = linear_term + args.method.q_coef * bellman

    loss_dict = {
        "recoil_q/linear_term": linear_term.item(),
        "recoil_q/q_policy_mean": q_policy_term.item(),
        "recoil_q/q_expert_mean": q_expert_term.item(),
        "recoil_q/bellman_loss": bellman.item(),
        "recoil_q/total": loss.item(),
    }
    return loss, loss_dict


def recoil_v_loss(agent, batch):
    """Eq. 12: E_{mix}[exp((Q - V)/tau) + (Q - V)/tau] (Q is detached)."""
    args = agent.args
    obs, _next_obs, action, _done, _is_expert = _split_batch(batch)

    with torch.no_grad():
        Q_mix = agent.critic(obs, action)

    V = agent.value_net(obs)
    tau = float(args.method.tau)
    diff = (Q_mix - V) / tau

    clip = float(args.method.awr_exp_clip)
    loss = (torch.exp(diff.clamp(max=clip)) + diff).mean()

    loss_dict = {
        "recoil_v/diff_mean": diff.mean().item(),
        "recoil_v/V_mean": V.mean().item(),
        "recoil_v/total": loss.item(),
    }
    return loss, loss_dict


def recoil_actor_loss(agent, batch):
    """Eq. 13: switchable AWR advantage (critic-only or original Q-V)."""
    args = agent.args
    obs, _next_obs, action, _done, _is_expert = _split_batch(batch)

    if _recoil_critic_only(agent):
        advantage = _target_advantage(agent, obs, action)
    else:
        with torch.no_grad():
            advantage = agent.critic(obs, action) - agent.value_net(obs)

    with torch.no_grad():
        weight = torch.exp(float(args.method.alpha) * advantage)
    weight = weight.clamp(max=float(args.method.awr_exp_clip))

    eps = _ACTION_LOGPROB_EPS
    safe_action = action.clamp(-1.0 + eps, 1.0 - eps)
    dist = agent.actor(obs)
    log_prob = dist.log_prob(safe_action).sum(-1, keepdim=True)

    loss = -(weight * log_prob).mean()

    loss_dict = {
        "recoil_pi/adv_mean": advantage.mean().item(),
        "recoil_pi/weight_mean": weight.mean().item(),
        "recoil_pi/weight_max": weight.max().item(),
        "recoil_pi/log_prob_mean": log_prob.mean().item(),
        "recoil_pi/total": loss.item(),
    }
    return loss, loss_dict


def recoil_update_critic(self, batch, logger, step):
    """Update Q_phi with the Eq. 11 semi-gradient loss (supports DoubleQCritic)."""
    obs, next_obs, action, _done, _is_expert = _split_batch(batch)

    with torch.no_grad():
        if _recoil_critic_only(self):
            next_V = _target_v_from_policy(self, next_obs, num_samples=1)
        else:
            next_V = self.value_net(next_obs)

    is_double_q = "DoubleQ" in self.args.q_net._target_
    if is_double_q:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1 = recoil_q_loss(self, current_Q1, next_V, batch)
        q2_loss, loss_dict2 = recoil_q_loss(self, current_Q2, next_V, batch)
        critic_loss = 0.5 * (q1_loss + q2_loss)
        loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict = recoil_q_loss(self, current_Q, next_V, batch)

    logger.log("train/critic_loss", critic_loss, step)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    return loss_dict


def recoil_update_value(self, batch, logger, step):
    v_loss, v_dict = recoil_v_loss(self, batch)
    logger.log("train/value_loss", v_loss, step)
    self.value_optimizer.zero_grad()
    v_loss.backward()
    self.value_optimizer.step()
    return v_dict


def recoil_update_actor(self, batch, logger, step):
    actor_loss, actor_dict = recoil_actor_loss(self, batch)
    logger.log("train/actor_loss", actor_loss, step)
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()
    if getattr(self, "she", False):
        self.scheduler.step()
    actor_dict["lr"] = self.actor_optimizer.param_groups[0]["lr"]
    return actor_dict


def recoil_update(self, policy_buffer, expert_buffer, logger, step):
    """One optimizer step for Q_phi (+ optional V_theta) and pi_psi."""
    args = self.args
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    batch = get_concat_samples(policy_batch, expert_batch, args)

    losses = recoil_update_critic(self, batch, logger, step)
    if not _recoil_critic_only(self):
        losses.update(recoil_update_value(self, batch, logger, step))

    if step % self.actor_update_frequency == 0:
        for _ in range(max(1, int(args.num_actor_updates))):
            actor_losses = recoil_update_actor(self, batch, logger, step)
        losses.update(actor_losses)

    if step % self.critic_target_update_frequency == 0:
        soft_update(self.critic_net, self.critic_target_net, self.critic_tau)

    return losses

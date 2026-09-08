"""
Copyright 2022 Div Garg. All rights reserved.

Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""
import math

import torch
import torch.nn.functional as F


def bellman_constraint(reward, args):
    """Elementwise implicit-reward constraint, shared by real and model data."""
    div = args.method.div
    if div == "hellinger":
        return torch.relu(reward - 1).square()
    if div == "kl":
        return torch.zeros_like(reward)
    if div == "kl2":
        return torch.relu(reward).square()
    if div == "chi":
        return torch.relu(-2 - reward).square()
    if div == "js":
        return torch.relu(reward - math.log(2.0)).square()
    high = torch.relu(reward - args.right)
    low = torch.relu(args.left - reward)
    return (F.smooth_l1_loss(high, torch.zeros_like(high), reduction='none')
            + F.smooth_l1_loss(low, torch.zeros_like(low), reduction='none'))

# Full IQ-Learn objective with other divergences and options
def iq_loss(agent, current_Q, current_v, next_v, batch, log_this_step=False,
            *, penalty_u, constraint_penalty):
    """Pure loss calculation; return loss, logs, and detached constraint mean."""
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward, done, is_expert = batch
    if args.cliptarget:
        next_v = torch.clip(next_v, args.left/(1-gamma), args.right/(1-gamma))
    loss_dict = {}
    # keep track of value of sampled expert/non-expert states
    expert_mask = is_expert.squeeze(1)
    non_expert_mask = ~expert_mask
    v0 = current_v.mean()
    if log_this_step:
        v0_expert = current_v[expert_mask].mean()
        v0_non_expert = current_v[non_expert_mask].mean()
        loss_dict['v0_expert'] = v0_expert.item()
        loss_dict['v0_non_expert'] = v0_non_expert.item()
    # loss_dict['v0_non_expert_std'] = current_v[non_expert_mask].std(unbiased=False).item()
    # loss_dict['Q_expert'] = current_Q[expert_mask].item()
    # loss_dict['Q_non_expert'] = current_Q[non_expert_mask].item()
    # loss_dict['v0_gap_expert_minus_non_expert'] = (v0 - v0_non_expert).item()

    #  calculate 1st term for IQ loss
    #  -E_(ρ_expert)[Q(s, a) - γV(s')]
    if args.usereal:
        y = (1 - done) * gamma * agent.getV(next_obs)
    else:
        y = (1-done) * gamma * next_v
    reward = (current_Q - y)[is_expert]
    if log_this_step:
        loss_dict['expert_reward'] = reward.mean().item()
        loss_dict['non_expert_reward'] = (current_Q - y)[~is_expert].mean().item()

    constraint_mean = None
    if penalty_u is not None:
        penalty_u = penalty_u.detach()
    if log_this_step:
        # loss_dict['penalty'] = penalty_u.mean().item() if penalty_u is not None else 0.0
        if penalty_u is not None:
            loss_dict['penalty_expert'] = penalty_u[expert_mask].mean().item()
            loss_dict['penalty_non_expert'] = penalty_u[non_expert_mask].mean().item()

    if penalty_u is not None:
        reward = reward + penalty_u[is_expert]
        
    with torch.no_grad():
        # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
        if args.method.div == "hellinger":
            phi_grad = 1/(1+reward)**2
        elif args.method.div == "kl":
            # original dual form for kl divergence (sub optimal)
            phi_grad = torch.exp(-reward-1)
        elif args.method.div == "kl2":
            # biased dual form for kl divergence
            phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
        elif args.method.div == "kl_fix":
            # our proposed unbiased form for fixing kl divergence
            phi_grad = torch.exp(-reward)
        elif args.method.div == "js":
            # jensen–shannon
            phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
        else:
            phi_grad = 1
    loss = -(phi_grad * reward).mean()
    if log_this_step:
        loss_dict['softq_loss'] = loss.item()

    # calculate 2nd term for IQ loss, we show different sampling strategies
    if args.method.loss == "value_expert":
        # sample using only expert states (works offline)
        # E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (current_v - y)[is_expert].mean()
        loss += value_loss
        if log_this_step:
            loss_dict['value_loss'] = value_loss.item()

    elif args.method.loss == "value":
        # sample using expert and policy states (works online)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = (current_v - y).mean()
        loss += args.value_ratio*value_loss
        if log_this_step:
            loss_dict['value_loss'] = value_loss.item()

    elif args.method.loss == "value_supplement":
        # sample using expert and policy states (works online)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = (current_v - y)[~is_expert].mean()
        loss += args.value_ratio*value_loss
        if log_this_step:
            loss_dict['value_loss'] = value_loss.item()

    elif args.method.loss == "v0":
        # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
        # (1-γ)E_(ρ0)[V(s0)]
        v0_loss = (1 - gamma) * v0
        loss += v0_loss
        if log_this_step:
            loss_dict['v0_loss'] = v0_loss.item()

    # alternative sampling strategies for the sake of completeness but are usually suboptimal in practice
    # elif args.method.loss == "value_policy":
    #     # sample using only policy states
    #     # E_(ρ)[V(s) - γV(s')]
    #     value_loss = (current_v - y)[~is_expert].mean()
    #     loss += value_loss
    #     loss_dict['value_policy_loss'] = value_loss.item()

    # elif args.method.loss == "value_mix":
    #     # sample by weighted combination of expert and policy states
    #     # E_(ρ)[Q(s,a) - γV(s')]
    #     w = args.method.mix_coeff
    #     value_loss = (w * (current_v - y)[is_expert] +
    #                   (1-w) * (current_v - y)[~is_expert]).mean()
    #     loss += value_loss
    #     loss_dict['value_loss'] = value_loss.item()

    else:
        raise ValueError(f'This sampling method is not implemented: {args.method.loss}')

    if args.method.grad_pen:
        # add a gradient penalty to loss (Wasserstein_1 metric)
        gp_loss = agent.critic_net.grad_pen(obs[is_expert.squeeze(1), ...],
                                            action[is_expert.squeeze(1), ...],
                                            obs[~is_expert.squeeze(1), ...],
                                            action[~is_expert.squeeze(1), ...],
                                            args.method.lambda_gp)
        if log_this_step:
            loss_dict['gp_loss'] = gp_loss.item()
        loss += gp_loss

    if args.method.div == "chi" or args.method.chi:  # TODO: Deprecate method.chi argument for method.div
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        
        if penalty_u is not None:
            reward = reward + penalty_u

        chi2_loss = 1/(4 * args.method.alpha) * (reward**2)[is_expert].mean()
        loss += chi2_loss
        if log_this_step:
            loss_dict['chi2_loss'] = chi2_loss.item()

    if args.method.regularize:
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
        y = (1 - done) * gamma * next_v
        
        reward = current_Q - y
        # reward = (current_Q - y)[~is_expert]
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
        if log_this_step:
            loss_dict['regularize_loss'] = chi2_loss.item()
    # else:
    #     y = (1 - done) * gamma * next_v
    #     reward = current_Q - y
    #     loss_dict['policy_reward'] = reward[~is_expert].mean().item()
    #     bellman_restrict = args.penalty*(torch.relu(args.left - reward)**2 + torch.relu(reward - args.right)**2).mean()
        
    #     loss += bellman_restrict
    #     loss_dict['bellman_restirct'] = bellman_restrict.item()

    if args.method.constrain:
        # for Bellman constrain
        y = (1 - done) * gamma * next_v

        reward = current_Q - y

        if penalty_u is not None:
            reward = reward + penalty_u

        constrain_loss = bellman_constraint(reward, args)

        constraint_mean = constrain_loss.mean()

        penalty = constraint_penalty.detach()

        loss += (penalty * constrain_loss).mean()
        # loss += (penalty * constrain_loss)[expert_mask].mean()

        if log_this_step:
            loss_dict['constrain_loss'] = (penalty * constrain_loss).mean().item()
            loss_dict['constrain_loss_expert'] = (penalty * constrain_loss)[expert_mask].mean().item()
            loss_dict['constrain_loss_non_expert'] = (penalty * constrain_loss)[~expert_mask].mean().item()
            loss_dict['constrain_loss_non_expert_positive'] = (torch.relu(reward - args.right))[~expert_mask].mean().item()
            loss_dict['penalty_alpha'] = float(penalty.item())

        # non_expert_constrain_loss = (penalty * constrain_loss)[~expert_mask].mean()
        # critic_params = [p for p in agent.critic.parameters() if p.requires_grad]
        # non_expert_grads = torch.autograd.grad(
        #     non_expert_constrain_loss,
        #     critic_params,
        #     retain_graph=True,
        #     create_graph=False,
        #     allow_unused=True,
        #     )
        # grad_norms_sup = [g.detach().norm() for g in non_expert_grads if g is not None]
        # if grad_norms_sup:
        #     grad_norms_sup = torch.stack(grad_norms_sup)
        #     loss_dict['constrain_grad_non_expert_mean'] = grad_norms_sup.mean().item()
        #     loss_dict['constrain_grad_non_expert_var'] = grad_norms_sup.var(unbiased=False).item()
        #     loss_dict['constrain_grad_non_expert_max'] = grad_norms_sup.max().item()
        #     loss_dict['constrain_grad_non_expert_total_norm'] = torch.linalg.vector_norm(grad_norms_sup).item()

        # expert_constrain_loss = (penalty * constrain_loss)[expert_mask].mean()
        # critic_params = [p for p in agent.critic.parameters() if p.requires_grad]
        # expert_grads = torch.autograd.grad(
        #     expert_constrain_loss,
        #     critic_params,
        #     retain_graph=True,
        #     create_graph=False,
        #     allow_unused=True,
        #     )
        # grad_norms = [g.detach().norm() for g in expert_grads if g is not None]
        # if grad_norms:
        #     grad_norms = torch.stack(grad_norms)
        #     loss_dict['constrain_grad_expert_mean'] = grad_norms.mean().item()
        #     loss_dict['constrain_grad_expert_var'] = grad_norms.var(unbiased=False).item()
            # loss_dict['constrain_grad_expert_max'] = grad_norms.max().item()
            # loss_dict['constrain_grad_expert_total_norm'] = torch.linalg.vector_norm(grad_norms).item()



    # # CQL penalty for continuous actions (sac)
    # if args.method.cql and hasattr(agent, "cqlV"):
        
    #     expert_mask = is_expert.squeeze(-1).bool()  # [B]
    #     cql_temp = getattr(args.method, "cql_temp", 1.0)

    #     # term1: E_{s~D}[ tau * logsumexp(Q(s,·)/tau) ], use all batch states
    #     # iq.py (continuous only)
    #     term1 = agent.cqlV(
    #         obs, agent.critic_net,
    #         num_random=args.method.cql_n_actions,
    #         temp=cql_temp
    #     )

    #     # term2: E_{(s,a)~expert}[Q(s,a)], use expert actions only
    #     term2 = current_Q[expert_mask].mean() if expert_mask.any() else current_Q.new_tensor(0.0)

    #     cql_loss = args.method.cql_alpha * (term1 - term2)

    #     loss += cql_loss
    #     loss_dict["cql_loss"] = cql_loss.item()
        


    if log_this_step:
        loss_dict['total_loss'] = loss.item()
    return loss, loss_dict, constraint_mean.detach() if constraint_mean is not None else None


def _compute_dynamics_penalty(agent, batch, *, shared_noise=True):
    """Compute c γ (1-done) Std_i(E_M[min_k Q_ψ_k^-(s',a')]).

    Set shared_noise=False only to diagnose the former independent sampler.

    Runs entirely under ``torch.no_grad`` so gradients do not flow through the
    dynamics ensemble, target critic or current actor sampling. Returns a
    tensor of shape [B, 1] aligned with ``(current_Q - y)``.

    """
    args = agent.args
    obs, _next_obs, action, _r, done, is_expert = batch

    if not hasattr(agent, "dynamics_ensemble"):
        raise RuntimeError(
            "method.uncertainty=True but agent.dynamics_ensemble is not set. "
            "Make sure the training entry point loads the ensemble checkpoint."
        )
    if getattr(agent, "actor", None) is None:
        # SoftQ (discrete actions) path is not supported here.
        return torch.zeros_like(obs[:, :1])

    ens = agent.dynamics_ensemble
    M = int(args.method.penalty_M)
    B = obs.size(0)
    if M < 1:
        raise ValueError("penalty_M must be at least 1")

    with torch.no_grad():
        # Pair samples across members, independently across batch/sample indices.
        dynamics_noise = None
        if shared_noise:
            dynamics_noise = torch.randn(B, M, ens.effective_obs_dim,
                                         device=obs.device, dtype=obs.dtype)
        next_states = ens.sample_next_ensemble(obs, action, M=M, noise=dynamics_noise)
        N = next_states.size(1)
        if N < 1 or N != int(args.method.penalty_N):
            raise ValueError("Sampled ensemble size must match penalty_N and be positive")
        flat_s = next_states.reshape(B * N * M, -1)
        flat_noise = None
        if shared_noise:
            actor_noise = torch.randn(B, M, action.size(-1), device=obs.device, dtype=obs.dtype)
            flat_noise = actor_noise.unsqueeze(1).expand(B, N, M, -1).reshape(B * N * M, -1)
        flat_a, _, _ = agent.actor.sample(flat_s, noise=flat_noise)

        critic_target = getattr(agent, "critic_target", None)
        if critic_target is None:
            critic_target = agent.critic  # fallback if no target net
        try:
            q1, q2 = critic_target(flat_s, flat_a, both=True)
            q_min = torch.min(q1, q2)
        except TypeError:
            # Critic does not expose ``both`` -> fall back to a single Q head.
            q_min = critic_target(flat_s, flat_a)

        q_min = q_min.view(B, N, M)
        per_member = q_min.mean(dim=2)  # E_M over samples: [B, N]
        # unbiased=False avoids NaN when N == 1.
        u = per_member.std(dim=1, unbiased=False, keepdim=True)  # [B, 1]
        u = agent.gamma * (1 - done.to(u).reshape(B, 1)) * u

    penalty_full = float(args.method.penalty_coef) * u  # [B, 1]
    return penalty_full


def synthetic_done(next_obs, env_name):
    """Observable termination rules for the repository's standard MuJoCo-v2 tasks.

    Hopper observations clip velocities, so its full simulator-state health
    check cannot be reconstructed exactly. Time-limit truncations bootstrap.
    Unknown tasks fail instead of borrowing done from a different action.
    """
    finite = torch.isfinite(next_obs).all(dim=-1, keepdim=True)
    height = next_obs[..., :1]
    angle = next_obs[..., 1:2]
    if env_name == 'Hopper-v2':
        healthy = ((height > 0.7) & (angle.abs() < 0.2)
                   & (next_obs[..., 1:].abs() < 100).all(dim=-1, keepdim=True))
    elif env_name == 'Walker2d-v2':
        healthy = (height > 0.8) & (height < 2.0) & (angle.abs() < 1.0)
    elif env_name == 'Ant-v2':
        healthy = (height >= 0.2) & (height <= 1.0)
    elif env_name == 'HalfCheetah-v2':
        healthy = torch.ones_like(finite)
    else:
        raise ValueError(f'Synthetic Bellman constraint has no termination rule for {env_name!r}')
    return ~(finite & healthy)


def validate_synthetic_config(agent):
    """Fail early on unsupported synthetic-constraint configurations."""
    args = agent.args
    method = args.method
    if not bool(getattr(method, 'synthetic_constrain', False)):
        return
    if method.type != 'iq' or not method.constrain:
        raise ValueError('synthetic_constrain requires method.type=iq and method.constrain=True')
    if getattr(agent, 'actor', None) is None or getattr(agent, 'critic_target', None) is None:
        raise ValueError('synthetic_constrain requires a continuous actor and target critic')
    coef = float(method.synthetic_coef)
    if not math.isfinite(coef) or coef < 0 or int(method.synthetic_warmup_steps) < 0:
        raise ValueError('synthetic_coef must be finite/nonnegative and warmup steps nonnegative')
    if int(method.synthetic_M) < 1 or int(method.penalty_N) < 1:
        raise ValueError('Synthetic sampling requires positive synthetic_M and penalty_N')
    if not callable(getattr(agent, 'synthetic_done_fn', None)):
        env_name = getattr(getattr(args, 'env', None), 'name', None)
        synthetic_done(torch.zeros(1, 2), env_name)


def synthetic_iq_loss(agent, obs, step, log_this_step=False):
    """Fresh one-step actor/model samples; only the online critic gets gradients.

    This auxiliary loss is independent of the real-data dual penalty. Model
    predictions are averaged in value space before applying the constraint.
    """
    args = agent.args
    method = args.method
    if not bool(getattr(method, 'synthetic_constrain', False)):
        return obs.new_zeros(()), {}
    coef = float(method.synthetic_coef)
    warmup = int(method.synthetic_warmup_steps)
    if warmup:
        coef *= min(1.0, max(0.0, float(step) / warmup))
    # Emit the same diagnostic keys at the first logged warmup step so CSV
    # loggers establish a stable schema. Non-logging zero-weight steps are free.
    if coef == 0 and not log_this_step:
        return obs.new_zeros(()), {}

    ens = agent.dynamics_ensemble
    B, M = obs.size(0), int(method.synthetic_M)
    with torch.no_grad():
        obs = obs.detach()
        action = agent.actor.sample(obs)[0]
        dynamics_noise = torch.randn(B, M, ens.effective_obs_dim,
                                     device=obs.device, dtype=obs.dtype)
        next_states = ens.sample_next_ensemble(obs, action, M=M, noise=dynamics_noise)
        N = next_states.size(1)
        if N != int(method.penalty_N):
            raise ValueError('Sampled ensemble size must match penalty_N')
        flat_s = next_states.reshape(B * N * M, -1)
        if not torch.isfinite(flat_s).all():
            raise ValueError('Dynamics produced non-finite synthetic next states')
        done_fn = getattr(agent, 'synthetic_done_fn', None)
        model_done = (done_fn(flat_s) if done_fn is not None
                      else synthetic_done(flat_s, args.env.name))
        continuation = (~model_done).to(obs)
        actor_noise = torch.randn(B, M, action.size(-1), device=obs.device, dtype=obs.dtype)
        flat_noise = actor_noise.unsqueeze(1).expand(B, N, M, -1).reshape(B * N * M, -1)
        next_action, log_prob, _ = agent.actor.sample(flat_s, noise=flat_noise)
        if 'DoubleQ' in args.q_net._target_:
            next_q1, next_q2 = agent.critic_target(flat_s, next_action, both=True)
            next_q = torch.minimum(next_q1, next_q2)
        else:
            next_q = agent.critic_target(flat_s, next_action)
        next_v = next_q - agent.alpha.detach() * log_prob
        if args.cliptarget:
            next_v = next_v.clamp(args.left / (1 - agent.gamma),
                                  args.right / (1 - agent.gamma))
        target = agent.gamma * (continuation * next_v).view(B, N, M).mean((1, 2)).unsqueeze(1)

        # Match the existing Q-based uncertainty definition, now evaluated on
        # actor actions, with continuation recomputed for each model transition.
        member_q = (continuation * next_q).view(B, N, M).mean(2)
        uncertainty = agent.gamma * member_q.std(1, unbiased=False, keepdim=True)
        penalty_u = (float(method.penalty_coef) * uncertainty
                     if method.uncertainty else torch.zeros_like(uncertainty))

    if 'DoubleQ' in args.q_net._target_:
        heads = agent.critic(obs, action, both=True)
    else:
        heads = (agent.critic(obs, action),)
    violations = torch.stack([bellman_constraint(q - target + penalty_u, args)
                              for q in heads])
    loss = coef * violations.mean()
    logs = {}
    if log_this_step:
        logs = {
            'synthetic/constrain_loss': loss.item(),
            'synthetic/violation': violations.mean().item(),
            'synthetic/coef': coef,
            'synthetic/uncertainty': uncertainty.mean().item(),
            'synthetic/q': torch.stack(heads).mean().item(),
            'synthetic/terminal_fraction': (1 - continuation).mean().item(),
        }
    return loss, logs


def prepare_iq_step(agent, batch):
    """Compute one detached Gamma and snapshot the constraint weight for this step."""
    penalty_u = (_compute_dynamics_penalty(agent, batch)
                 if agent.args.method.uncertainty else None)
    penalty = batch[0].new_tensor(float(agent.args.penalty))
    return penalty_u, penalty


def update_iq_penalty(agent, constraint_means):
    """Update the dual weight once, after the critic, using all heads equally."""
    method = agent.args.method
    if not method.constrain or not bool(getattr(method, "penalty_auto", False)):
        return
    values = [value.detach() for value in constraint_means if value is not None]
    if not values:
        raise ValueError("Automatic penalty update requires constraint statistics")
    constraint_mean = torch.stack(values).mean()
    if not hasattr(agent, "log_penalty"):
        agent.log_penalty = constraint_mean.new_tensor(
            math.log(max(float(agent.args.penalty), 1e-8)), requires_grad=True)
        agent.penalty_optimizer = torch.optim.Adam(
            [agent.log_penalty], lr=float(getattr(method, "penalty_lr", 0.01)))
    target = float(getattr(method, "penalty_target", 0.0))
    penalty_loss = -agent.log_penalty * (constraint_mean - target)
    agent.penalty_optimizer.zero_grad()
    penalty_loss.backward()
    agent.penalty_optimizer.step()
    with torch.no_grad():
        penalty = agent.log_penalty.exp().clamp(
            float(getattr(method, "penalty_min", 0.0)),
            float(getattr(method, "penalty_max", 1e6)))
        agent.log_penalty.copy_(penalty.clamp_min(1e-8).log())
        agent.args.penalty = float(penalty.item())

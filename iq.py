"""
Copyright 2022 Div Garg. All rights reserved.

Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""
import math

import torch
import torch.nn.functional as F

# Full IQ-Learn objective with other divergences and options
def iq_loss(agent, current_Q, current_v, next_v, batch):
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
    loss_dict['expert_reward'] = reward.mean().item()
    loss_dict['non_expert_reward'] = (current_Q - y)[~is_expert].mean().item()

    penalty_u = _compute_dynamics_penalty(agent, batch) if args.method.penalty else None
    loss_dict['penalty'] = penalty_u.mean().item() if penalty_u is not None else 0.0
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
    loss_dict['softq_loss'] = loss.item()

    # calculate 2nd term for IQ loss, we show different sampling strategies
    if args.method.loss == "value_expert":
        # sample using only expert states (works offline)
        # E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (current_v - y)[is_expert].mean()
        loss += value_loss
        loss_dict['value_loss'] = value_loss.item()

    elif args.method.loss == "value":
        # sample using expert and policy states (works online)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = (current_v - y).mean()
        loss += args.value_ratio*value_loss
        loss_dict['value_loss'] = value_loss.item()

    elif args.method.loss == "value_supplement":
        # sample using expert and policy states (works online)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = (current_v - y)[~is_expert].mean()
        loss += args.value_ratio*value_loss
        loss_dict['value_loss'] = value_loss.item()

    elif args.method.loss == "v0":
        # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
        # (1-γ)E_(ρ0)[V(s0)]
        v0_loss = (1 - gamma) * v0
        loss += v0_loss
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
        loss_dict['gp_loss'] = gp_loss.item()
        loss += gp_loss

    if args.method.div == "chi" or args.method.chi:  # TODO: Deprecate method.chi argument for method.div
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        # print(args.method.alpha)
        # exit()
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2)[is_expert].mean()
        loss += chi2_loss
        loss_dict['chi2_loss'] = chi2_loss.item()

    if args.method.regularize:
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
        y = (1 - done) * gamma * next_v
        
        reward = current_Q - y
        # reward = (current_Q - y)[~is_expert]
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
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

        if args.method.div == "hellinger":
            constrain_loss = (torch.relu(reward - 1))**2
            # phi_grad = 1/(1+reward)**2
        elif args.method.div == "kl":
            # original dual form for kl divergence (sub optimal)
            constrain_loss = torch.zeros_like(reward)
        elif args.method.div == "kl2":
            # biased dual form for kl divergence
            constrain_loss = (torch.relu(reward))**2
        elif args.method.div == "chi":
            constrain_loss = (torch.relu(-2 - reward))**2
        elif args.method.div == "js":
            # jensen–shannon
            constrain_loss = (torch.relu(reward - math.log(2.0)))**2
        else:
            constrain_loss = (torch.relu(reward - args.right))**2 + (torch.relu(args.left - reward))**2

            # constrain_loss = torch.relu(reward - args.right) + torch.relu(args.left - reward)

            # diff_high = torch.relu(reward - args.right)
            # diff_low = torch.relu(args.left - reward)
            # constrain_loss =F.smooth_l1_loss(diff_high, torch.zeros_like(diff_high), reduction='none')+F.smooth_l1_loss(diff_low, torch.zeros_like(diff_high), reduction='none')
           
        constraint_mean = constrain_loss.mean()

        penalty_auto = bool(getattr(args.method, "penalty_auto", False))
        target_constraint = float(getattr(args.method, "penalty_target", 0.0))
        penalty_lr = float(getattr(args.method, "penalty_lr", 0.01))
        penalty_min = float(getattr(args.method, "penalty_min", 0.0))
        penalty_max = float(getattr(args.method, "penalty_max", 1e6))

        if penalty_auto:
            if not hasattr(agent, "log_penalty"):
                init_penalty = float(args.penalty)
                log_penalty = torch.tensor(math.log(max(init_penalty, 1e-8)), device=constraint_mean.device)
                log_penalty.requires_grad_(True)
                agent.log_penalty = log_penalty
                agent.penalty_optimizer = torch.optim.Adam([agent.log_penalty], lr=penalty_lr)

            penalty_loss = -(agent.log_penalty * (constraint_mean - target_constraint).detach())
            agent.penalty_optimizer.zero_grad()
            penalty_loss.backward()
            agent.penalty_optimizer.step()

            with torch.no_grad():
                penalty = agent.log_penalty.exp().clamp(penalty_min, penalty_max)
                agent.log_penalty.copy_(torch.log(penalty.clamp_min(1e-8)))
                args.penalty = float(penalty.item())
        else:
            penalty = torch.tensor(float(args.penalty), device=constraint_mean.device)

        loss += (penalty * constrain_loss).mean()
        # loss += (penalty * constrain_loss)[expert_mask].mean()

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
        


    loss_dict['total_loss'] = loss.item()
    return loss, loss_dict


def _compute_dynamics_penalty(agent, batch):
    """Compute the uncertainty penalty U(s,a) = γ · Std_i(E_M[min_k Q_ψ_k^-(s',a')]).

    Runs entirely under ``torch.no_grad`` so gradients do not flow through the
    dynamics ensemble, target critic or current actor sampling. Returns a 1D
    tensor aligned with ``(current_Q - y)``.

    Sign convention: larger uncertainty -> larger penalty subtracted from the
    implicit reward, i.e. we return ``coef * U`` so the caller can
    keep the natural ``reward = reward + penalty`` formulation.
    """
    args = agent.args
    obs, _next_obs, action, _r, _done, is_expert = batch

    if not hasattr(agent, "dynamics_ensemble"):
        raise RuntimeError(
            "method.penalty=True but agent.dynamics_ensemble is not set. "
            "Make sure the ensemble checkpoint is loaded in train_iq.py."
        )
    if getattr(agent, "actor", None) is None:
        # SoftQ (discrete actions) path is not supported here.
        return torch.zeros_like(obs[:, :1])

    ens = agent.dynamics_ensemble
    N = int(args.method.penalty_N)
    M = int(args.method.penalty_M)
    B = obs.size(0)

    with torch.no_grad():
        next_states = ens.sample_next_ensemble(obs, action, M=M)  # [B, N, M, obs_dim]
        flat_s = next_states.reshape(B * N * M, -1)
        flat_a, _, _ = agent.actor.sample(flat_s)

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
        u = args.gamma * u

    penalty_full = float(args.method.penalty_coef) * u  # [B, 1]
    return penalty_full

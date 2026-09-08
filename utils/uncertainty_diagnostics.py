"""Fixed-batch Monte Carlo diagnostics for an already loaded IQ agent.

No optimizer steps are performed. The caller's RNG states, module training flags,
and penalty_M are restored, including when estimation raises an exception.
"""
import torch

from iq import _compute_dynamics_penalty


@torch.no_grad()
def diagnose_uncertainty(agent, batch, sample_counts=(10, 50, 100, 500),
                         repeats=50, seed=0):
    """Return JSON-compatible summaries, separated by expert/non-expert rows.

    Freeze a checkpoint and batch before calling. The largest-M shared-noise
    repeated mean is a numerical reference, not ground truth. `repeat_std` is
    computed per state across repeats, then averaged over each group.
    Supports CPU/CUDA; this routine never changes the training default M.
    """
    counts = sorted(set(int(m) for m in sample_counts))
    if not counts or counts[0] < 1 or repeats < 2:
        raise ValueError("Use positive sample counts and at least two repeats")
    device = batch[0].device
    if device.type not in ('cpu', 'cuda'):
        raise ValueError("RNG-isolated diagnostics currently support CPU and CUDA")
    modules = [agent.dynamics_ensemble, agent.actor,
               getattr(agent, 'critic_target', None) or agent.critic]
    modes = {module: module.training for root in modules for module in root.modules()}
    original_m = agent.args.method.penalty_M
    samples = {}
    # Fork all CUDA devices so manual_seed cannot affect a caller's other device.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        for module in modules:
            module.eval()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for shared in (False, True):
                for m in counts:
                    agent.args.method.penalty_M = m
                    samples[(shared, m)] = torch.stack([
                        _compute_dynamics_penalty(agent, batch, shared_noise=shared)
                        .squeeze(-1).cpu() for _ in range(repeats)
                    ])
    finally:
        agent.args.method.penalty_M = original_m
        for module, training in modes.items():
            module.training = training

    expert = batch[5].squeeze(-1).bool().cpu()
    terminal = batch[4].reshape(-1).bool().cpu()
    groups = {'all': torch.ones_like(expert), 'expert': expert,
              'non_expert': ~expert, 'terminal': terminal, 'continuing': ~terminal}
    reference = samples[(True, counts[-1])].mean(0)
    rows = []
    for (shared, m), estimates in samples.items():
        per_state_mean = estimates.mean(0)
        per_state_std = estimates.std(0, unbiased=False)
        for group, mask in groups.items():
            if not mask.any():
                continue
            rows.append(dict(
                sampling='shared' if shared else 'independent', M=m, group=group,
                states=int(mask.sum()), gamma_mean=float(per_state_mean[mask].mean()),
                gamma_squared_mean=float(estimates[:, mask].square().mean()),
                repeat_std=float(per_state_std[mask].mean()),
                mean_absolute_difference_from_reference=float(
                    (per_state_mean[mask] - reference[mask]).abs().mean()),
            ))
    return dict(seed=seed, repeats=repeats, reference_M=counts[-1], results=rows)

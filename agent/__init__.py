import gym
from agent.sac import SAC
from agent.softq import SoftQ


def make_agent(env, args):
    obs_dim = env.observation_space.shape[0]
    method_type = getattr(args.method, "type", None)

    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        if method_type == "recoil":
            raise NotImplementedError(
                "ReCOIL currently only supports continuous action spaces."
            )
        print('--> Using Soft-Q agent')
        action_dim = env.action_space.n
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = SoftQ(obs_dim, action_dim, args.train.batch, args)
    else:
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        if method_type == "recoil":
            print('--> Using ReCOIL agent')
            from agent.recoil import ReCOIL
            agent = ReCOIL(obs_dim, action_dim, action_range, args.train.batch, args)
        else:
            print('--> Using SAC agent')
            agent = SAC(obs_dim, action_dim, action_range, args.train.batch, args)

    return agent

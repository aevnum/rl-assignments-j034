from __future__ import annotations

from gridworld import make_default_grid, sample_next_state_and_reward
from rtdp import RTDP, RTDPConfig, LinearDecay
from mcts import MCTS, MCTSConfig


def run_rtdp():
    env = make_default_grid()
    cfg = RTDPConfig(
        gamma=0.95,
        episodes=50,
        max_steps=1000,
        epsilon_schedule=LinearDecay(start=0.5, end=0.05, steps=50),
    )
    agent = RTDP(env, cfg)
    agent.run()  # Will raise NotImplementedError until students implement


def run_mcts():
    env = make_default_grid()
    cfg = MCTSConfig(gamma=0.95, c_uct=1.4, rollouts=200, max_depth=200)
    agent = MCTS(env, cfg)
    episodes = 20
    for ep in range(episodes):
        s = env.initial_state()
        steps = 0
        total_reward = 0.0
        while not env.is_terminal(s) and steps < cfg.max_depth:
            a = agent.search(s)
            s_p, r = sample_next_state_and_reward(env, s, a, rng=agent.rng)
            total_reward += r
            s = s_p
            steps += 1
        print(f"MCTS Episode {ep+1}: steps={steps}, total_reward={total_reward:.2f}")


if __name__ == "__main__":
    # Choose one to test
    # run_rtdp()
    run_mcts()
    pass


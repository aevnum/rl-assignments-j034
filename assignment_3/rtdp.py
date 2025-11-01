from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from gridworld import MDP, State, Action, sample_next_state_and_reward


@dataclass
class LinearDecay:
    start: float
    end: float
    steps: int

    def value(self, t: int) -> float:
        if t <= 0:
            return float(self.start)
        if t >= self.steps:
            return float(self.end)
        frac = t / float(self.steps)
        return float(self.start + frac * (self.end - self.start))


@dataclass
class RTDPConfig:
    gamma: float = 0.95
    episodes: int = 50
    max_steps: int = 1_000
    epsilon_schedule: LinearDecay | None = None


class RTDP:
    def __init__(self, mdp: MDP, cfg: RTDPConfig, rng=None, heuristic=None) -> None:
        self.mdp = mdp
        self.cfg = cfg
        self.rng = rng
        self.heuristic = heuristic
        self.V: Dict[State, float] = {}

        if self.rng is None:
            import random

            self.rng = random.Random(0)

    def value(self, s: State) -> float:
        if s not in self.V:
            self.V[s] = float(self.heuristic(s) if self.heuristic else 0.0)
        return self.V[s]

    def bellman_backup(self, s: State) -> float:
        actions = self.mdp.actions(s)
        if not actions:
            self.V[s] = 0.0
            return 0.0

        # Compute Q(s,a) for each action
        q_values = []
        for a in actions:
            transitions = self.mdp.transitions(s, a)
            q = 0.0
            for t in transitions:
                s_p = t.next_state
                r = t.reward
                p = t.probability
                q += p * (r + self.cfg.gamma * self.value(s_p))
            q_values.append(q)
        v = max(q_values)
        self.V[s] = v
        return v

    def select_action(self, s: State, epsilon: float) -> Action:
        actions = list(self.mdp.actions(s))
        assert actions
        # Epsilon-greedy over one-step lookahead Q(s,a)
        if self.rng.random() < epsilon:
            return self.rng.choice(actions)
        # Compute Q(s,a) for all actions
        q_values = []
        for a in actions:
            transitions = self.mdp.transitions(s, a)
            q = 0.0
            for t in transitions:
                s_p = t.next_state
                r = t.reward
                p = t.probability
                q += p * (r + self.cfg.gamma * self.value(s_p))
            q_values.append(q)
        max_idx = max(range(len(q_values)), key=lambda i: q_values[i])
        return actions[max_idx]

    def run(self) -> None:
        episodes = self.cfg.episodes
        for ep in range(episodes):
            s = self.mdp.initial_state()
            steps = 0
            total_reward = 0.0
            epsilon = self.cfg.epsilon_schedule.value(ep) if self.cfg.epsilon_schedule else 0.0
            while (not self.mdp.is_terminal(s)) and (steps < self.cfg.max_steps):
                self.bellman_backup(s)
                a = self.select_action(s, epsilon)
                s_p, r = sample_next_state_and_reward(self.mdp, s, a, rng=self.rng)
                total_reward += r
                s = s_p
                steps += 1
            print(f"Episode {ep+1}: steps={steps}, total_reward={total_reward:.2f}")


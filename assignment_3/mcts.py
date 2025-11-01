from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from gridworld import MDP, State, Action, sample_next_state_and_reward


@dataclass
class MCTSConfig:
    gamma: float = 0.95
    c_uct: float = 1.4
    rollouts: int = 200
    max_depth: int = 200


class Node:
    def __init__(self, state: State, parent: Optional[Tuple["Node", Action]] = None) -> None:
        self.state = state
        self.parent = parent
        self.children: Dict[Action, Node] = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / float(self.visits)


class MCTS:
    def __init__(self, mdp: MDP, cfg: MCTSConfig, rng=None, heuristic=None) -> None:
        self.mdp = mdp
        self.cfg = cfg
        self.rng = rng
        self.heuristic = heuristic
        if self.rng is None:
            import random

            self.rng = random.Random(0)

    def search(self, root_state: State) -> Action:
        root = Node(root_state)
        for _ in range(self.cfg.rollouts):
            # --- Selection ---
            node = root
            path = []  # (node, action) pairs
            depth = 0
            while True:
                actions = list(self.mdp.actions(node.state))
                if not actions or depth >= self.cfg.max_depth:
                    break
                # If not fully expanded, expand
                untried = [a for a in actions if a not in node.children]
                if untried:
                    a = self.rng.choice(untried)
                    s_p, r = sample_next_state_and_reward(self.mdp, node.state, a, rng=self.rng)
                    child = Node(s_p, parent=(node, a))
                    node.children[a] = child
                    path.append((node, a))
                    node = child
                    break  # Expansion
                # UCT selection
                c = self.cfg.c_uct
                total_visits = sum(ch.visits for ch in node.children.values())
                def uct_score(n, N, q, c):
                    return q + c * math.sqrt(math.log(N + 1) / (1 + n))
                best_a = max(node.children.keys(), key=lambda a: uct_score(
                    node.children[a].visits,
                    total_visits,
                    node.children[a].q,
                    c))
                path.append((node, best_a))
                node = node.children[best_a]
                depth += 1

            # --- Rollout ---
            s = node.state
            total_reward = 0.0
            discount = 1.0
            for rollout_depth in range(self.cfg.max_depth - depth):
                if self.mdp.is_terminal(s):
                    break
                actions = list(self.mdp.actions(s))
                if not actions:
                    break
                a = self.rng.choice(actions)
                s_p, r = sample_next_state_and_reward(self.mdp, s, a, rng=self.rng)
                total_reward += discount * r
                discount *= self.cfg.gamma
                s = s_p

            # --- Backpropagation ---
            # Backpropagate from node up the path
            node_to_update = node
            reward = total_reward
            for parent, a in reversed(path):
                node_to_update.visits += 1
                node_to_update.value_sum += reward
                reward = parent.value_sum / parent.visits if parent.visits > 0 else 0.0
                node_to_update = parent
            # Update root
            root.visits += 1
            root.value_sum += total_reward

        # choose action with most visits
        best_a = None
        best_v = -1
        for a, ch in root.children.items():
            if ch.visits > best_v:
                best_v = ch.visits
                best_a = a
        if best_a is None:
            actions = list(self.mdp.actions(root_state))
            if not actions:
                raise RuntimeError("MCTS on terminal state")
            best_a = actions[0]
        return best_a


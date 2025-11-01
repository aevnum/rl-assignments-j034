## RTDP (decaying epsilon) + MCTS — Assignment

Your job: fill the sections marked `#YOUR CODE HERE` in `rtdp.py` and `mcts.py`.

### Tasks (do these):
1. RTDP: implement `bellman_backup` and the episode loop with decaying epsilon-greedy.
2. MCTS (UCT): implement one iteration (selection, expansion, rollout, backprop) and return the most visited action.
3. Run both on the default GridWorld and print steps-to-goal and total reward for at least 20 episodes/searches.
4. Briefly (3–5 sentences) compare RTDP vs MCTS behavior on this map.

- RTDP learns from its mistakes and gets better at finding the goal, like a kid who remembers the right path in a maze after trying a few times. MCTS tries to figure out the best move each time from the start, so sometimes it takes a long way and sometimes a short way, like a kid guessing which way to go every time without remembering what worked before. RTDP gets faster and smarter, but MCTS keeps guessing.

5. Optional: try different epsilon schedules and `c_uct` values and comment.

### Run
- Edit `main.py` and uncomment `run_rtdp()` or `run_mcts()`.
- `python main.py`

### Hints
- RTDP: `V[s] = max_a E[r + gamma V[s']]` using the provided model. Use epsilon-greedy over one-step lookahead Q(s,a).
- MCTS: UCT score `Q + c * sqrt(ln N / (1 + N_a))`. Discount returns in rollout/backprop.


import gymnasium as gym

# Environment: LunarLander-v2
# Goal/Objective: The goal is to land a lunar module safely on the moon's surface. The lander starts above the surface and must use its engines to control descent and landing without crashing or running out of fuel.
# Observation space: Box(8,) - continuous values representing the lander's x position, y position, x velocity, y velocity, angle, angular velocity, and boolean flags for left and right leg contact with ground
# Action space: Discrete(4) - 0: do nothing, 1: fire left engine, 2: fire main engine, 3: fire right engine
# Episode length: Typically up to 1000 steps, but can end earlier if landed or crashed
# What makes it challenging: Complex physics with gravity, thrust, and rotation. Need to balance multiple objectives (landing safely, conserving fuel, avoiding crash). Sparse rewards make learning difficult.

env = gym.make('LunarLander-v2', render_mode='human')
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Manual play mode: Use 0 (no action), 1 (left engine), 2 (main engine), 3 (right engine)
def get_user_action():
    while True:
        try:
            action = int(input("Action (0=none, 1=left, 2=main, 3=right): "))
            if action in [0, 1, 2, 3]:
                return action
            else:
                print("Invalid action. Enter 0, 1, 2, or 3.")
        except ValueError:
            print("Please enter a number (0, 1, 2, or 3).")

obs, info = env.reset()
done = False
steps = 0
total_reward = 0
while not done and steps < 1000:
    env.render()
    action = get_user_action()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    steps += 1

print(f"Episode length: {steps}")
print(f"Total reward: {total_reward}")

env.close()
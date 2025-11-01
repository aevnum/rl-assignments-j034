import gymnasium as gym

# Environment: MountainCar-v0
# Goal/Objective: The goal is to drive an underpowered car up a steep hill. The car starts at the bottom of a valley and must build momentum by swinging back and forth to reach the top of the hill on the right.
# Observation space: Box(2,) - continuous values representing the car's position (between -1.2 and 0.6) and velocity (between -0.07 and 0.07)
# Action space: Discrete(3) - 0: accelerate left, 1: do nothing, 2: accelerate right
# Episode length: Typically up to 200 steps, but can be longer if not solved
# What makes it challenging: The car doesn't have enough power to climb the hill directly from the bottom. It needs to learn to oscillate back and forth to build momentum, which requires understanding delayed rewards and exploration.

env = gym.make('MountainCar-v0', render_mode='human')
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Manual play mode: Use 0 (left), 1 (no action), 2 (right) keys to control the car
def get_user_action():
    while True:
        try:
            action = int(input("Action (0=left, 1=none, 2=right): "))
            if action in [0, 1, 2]:
                return action
            else:
                print("Invalid action. Enter 0, 1, or 2.")
        except ValueError:
            print("Please enter a number (0, 1, or 2).")

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
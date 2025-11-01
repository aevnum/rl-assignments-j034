import gymnasium as gym

# Environment: Acrobot-v1
# Goal/Objective: The goal is to swing up a double pendulum (acrobot) from its hanging position to an upright position where the end of the second link is above a certain height.
# Observation space: Box(6,) - continuous values representing the cosine and sine of the two joint angles, and their angular velocities
# Action space: Discrete(3) - 0: apply negative torque to joint 1, 1: apply no torque, 2: apply positive torque to joint 1
# Episode length: Typically up to 500 steps
# What makes it challenging: Underactuated system (only one actuator), requires precise timing and momentum to swing up. Sparse rewards until goal is reached.

env = gym.make('Acrobot-v1', render_mode='human')
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Manual play mode: Use 0 (negative torque), 1 (no torque), 2 (positive torque)
def get_user_action():
    while True:
        try:
            action = int(input("Action (0=neg torque, 1=none, 2=pos torque): "))
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
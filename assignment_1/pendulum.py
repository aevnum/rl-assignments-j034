import gymnasium as gym

# Environment: Pendulum-v1
# Goal/Objective: The goal is to balance a pendulum upright by applying torque to the joint. The pendulum starts in a random position, and the agent must keep it balanced.
# Observation space: Box(3,) - continuous values representing cosine of angle, sine of angle, and angular velocity
# Action space: Box(1,) - continuous torque value between -2.0 and 2.0
# Episode length: Typically up to 200 steps
# What makes it challenging: Continuous action space requires fine control. The system is unstable, and small errors can lead to failure. Rewards are based on angle and velocity, encouraging stability.

env = gym.make('Pendulum-v1', render_mode='human')
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Manual play mode: Enter a float for torque between -2.0 and 2.0
def get_user_action():
    while True:
        try:
            action = float(input("Action (torque, -2.0 to 2.0): "))
            if -2.0 <= action <= 2.0:
                return [action]
            else:
                print("Invalid action. Enter a value between -2.0 and 2.0.")
        except ValueError:
            print("Please enter a float value between -2.0 and 2.0.")

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
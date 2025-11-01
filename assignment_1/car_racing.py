import gymnasium as gym

# Environment: CarRacing-v2
# Goal/Objective: The goal is to drive a car around a racetrack as fast as possible without going off the track. The car must complete laps while staying on the road.
# Observation space: Box(96, 96, 3) - RGB image of the track and car, with values 0-255
# Action space: Box(3,) - continuous values for steering (-1 to 1), gas (0 to 1), brake (0 to 1)
# Episode length: Typically up to 1000 steps
# What makes it challenging: High-dimensional image observations require computer vision techniques. Continuous actions need precise control. The track has varying shapes, and off-track penalties make exploration risky.

env = gym.make('CarRacing-v2', render_mode='human')
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Manual play mode: Enter steering (-1 to 1), gas (0 to 1), brake (0 to 1) each step
def get_user_action():
    while True:
        try:
            steering = float(input("Steering (-1 to 1): "))
            gas = float(input("Gas (0 to 1): "))
            brake = float(input("Brake (0 to 1): "))
            if -1.0 <= steering <= 1.0 and 0.0 <= gas <= 1.0 and 0.0 <= brake <= 1.0:
                return [steering, gas, brake]
            else:
                print("Invalid input. Steering: -1 to 1, Gas: 0 to 1, Brake: 0 to 1.")
        except ValueError:
            print("Please enter valid float values.")

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
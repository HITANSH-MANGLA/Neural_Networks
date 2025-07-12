import argparse
import gym
import warnings

# Ignore NumPy bool8 deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run a Gym environment.")
    parser.add_argument('--input-env', dest='input_env', required=False,
                        default='cartpole',  # default bana diya
                        choices=['cartpole','mountaincar','pendulum','taxi','lake'],
                        help = 'specify the name of the environment to run')
    return parser

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    input_env = args.input_env

    name_map = {
        'cartpole': 'CartPole-v0',
        'mountaincar': 'MountainCar-v0',
        'pendulum': 'Pendulum-v0',
        'taxi': 'Taxi-v3',  # latest version
        'lake': 'FrozenLake-v1'  # v0 is deprecated
    }

    # Create the environment with render_mode
    env = gym.make(name_map[input_env], render_mode="human")
    observation, info = env.reset()  # latest gym returns (obs, info)

    for _ in range(1000):
        action = env.action_space.sample()
        result = env.step(action)

        # Unpack based on Gym's newer API (returns 5 values)
        observation, reward, terminated, truncated, info = result
        done = terminated or truncated

        if done:
            print("Episode done. Resetting environment.")
            observation, info = env.reset()

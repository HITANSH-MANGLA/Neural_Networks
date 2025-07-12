import argparse
import gym

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run a Gym environment.")
    parser.add_argument('--input-env', dest='input_env', required=True,
                        
                        choices=['cartpole', 'mountaincar', 'pendulum'],
                        help='specify the name of the environment to run')
    return parser

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    input_env = args.input_env

    name_map = {
        'cartpole': 'CartPole-v0',
        'mountaincar': 'MountainCar-v0',
        'pendulum': 'Pendulum-v0'
    }

    # Create the environment with render_mode
    env = gym.make(name_map[input_env], render_mode="human")
    for _ in range(20):
        observation, info = env.reset()
        for i in range(100):
            env.render()
            print(observation)

            action = env.action_space.sample()  # Random action
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(i + 1))
                break
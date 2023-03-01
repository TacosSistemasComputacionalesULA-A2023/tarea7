import gym
from agent import QLearning
from agent import DoubleQLearning
from threading import Lock
import os

NUM_EPISODES = 500

class Arguments:
    def __init__(self, env_name, algorithm, alpha, gamma, epsilon) -> None:
        self.env_name = env_name
        self.algorithm = algorithm
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        

def init_data(experiments_counter, expirements_num):
    global experiments, total_experiments
    total_experiments = expirements_num
    experiments = experiments_counter

def train(env, agent, episodes):
    total_reward = 0
    end_episode_reward = []
    for episode in range(episodes + 1):
        observation, _ = env.reset()
        terminated, truncated = False, False

        episode_reward = 0
        while not (terminated or truncated):
            action = agent.get_action(observation, "epsilon-greedy")
            new_observation, reward, terminated, truncated, _ = env.step(
                action)
            episode_reward += reward
            agent.update(observation, action,
                         new_observation, reward, terminated)
            observation = new_observation
        
        end_episode_reward.append(episode_reward)
        total_reward += episode_reward

    return total_reward, end_episode_reward


def run_training(arguments: Arguments):
    # Create the environment
    env = gym.make(arguments.env_name)

    # Run your reinforcement learning algorithm here
    total_reward = 0
    end_episode_reward = []
    if arguments.algorithm == 'Q-learning':
        # Run Q-learning algorithm
        agent = QLearning(
            env.observation_space.n,
            env.action_space.n,
            alpha=arguments.alpha,
            gamma=arguments.gamma,
            epsilon=arguments.epsilon
        )

        total_reward, end_episode_reward = train(env, agent, NUM_EPISODES)
        

    elif arguments.algorithm == 'DoubleQ-learning':
        # Run DoubleQ-learning algorithm
        agent = DoubleQLearning(
            env.observation_space.n,
            env.action_space.n,
            alpha=arguments.alpha,
            gamma=arguments.gamma,
            epsilon=arguments.epsilon
        )

        total_reward, end_episode_reward = train(env, agent, NUM_EPISODES)
        

    # Close the environment
    env.close()
    global experiments
    global total_experiments
    with experiments.get_lock():
        experiments.value -= 1
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f'Experiments left: {experiments.value}/{total_experiments}')

    # Add the results to the list
    return (arguments.env_name, arguments.algorithm, arguments.alpha, 
            arguments.gamma, arguments.epsilon, NUM_EPISODES, total_reward, end_episode_reward)

import gym
from agent import QLearning
from agent import DoubleQLearning

class Arguments:
    def __init__(self, env_name, algorithm, num_episodes, alpha, gamma, epsilon) -> None:
        self.env_name = env_name
        self.algorithm = algorithm
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon


def train(env, agent, episodes):
    total_reward = 0
    for episode in range(episodes + 1):
        print(episode)
        observation, _ = env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.get_action(observation, "epsilon-greedy")
            new_observation, reward, terminated, truncated, _ = env.step(
                action)
            total_reward += reward
            agent.update(observation, action,
                         new_observation, reward, terminated)
            observation = new_observation

    return total_reward


def run_training(arguments: Arguments):
    # Create the environment
    env = gym.make(arguments.env_name)

    # Run your reinforcement learning algorithm here
    total_reward = 0
    if arguments.algorithm == 'Q-learning':
        # Run Q-learning algorithm
        agent = QLearning(
            env.observation_space.n,
            env.action_space.n,
            alpha=arguments.alpha,
            gamma=arguments.gamma,
            epsilon=arguments.epsilon
        )

        total_reward = train(env, agent, arguments.num_episodes)
        agent.reset()

    elif arguments.algorithm == 'DoubleQ-learning':
        # Run DoubleQ-learning algorithm
        agent = DoubleQLearning(
            env.observation_space.n,
            env.action_space.n,
            alpha=arguments.alpha,
            gamma=arguments.gamma,
            epsilon=arguments.epsilon
        )

        total_reward = train(env, agent, arguments.num_episodes)
        agent.reset()

    # Close the environment
    env.close()

    # Add the results to the list
    return (arguments.env_name, arguments.algorithm, arguments.alpha, 
            arguments.gamma, arguments.epsilon, arguments.num_episodes, total_reward)

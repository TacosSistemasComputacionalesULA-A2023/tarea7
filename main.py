import gym
import csv
import gym_environments
from agent import QLearning
from agent import DoubleQLearning

def train(env, agent, episodes):
    total_reward = 0
    for episode in range(episodes + 1):
        print(episode)
        observation, _ = env.reset()
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            action = agent.get_action(observation, "epsilon-greedy")
            new_observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            agent.update(observation, action, new_observation, reward, terminated)
            observation = new_observation

    return total_reward

# Define the environment names
env_names = ["CliffWalking-v0", "Taxi-v3"]

# Define the range of values for alpha, gamma, and epsilon
alpha_range = [round(i * 0.1, 1) for i in range(1, 10)]
gamma_range = [round(i * 0.1, 1) for i in range(1, 10)]
epsilon_range = [round(i * 0.1, 1) for i in range(1, 10)]

# Define the reinforcement learning algorithms
algorithms = ['Q-learning', 'DoubleQ-learning']

# Define the range of values for the number of episodes
num_episodes_range = range(1, 501)

# Create a list to store the results
results = []

# Iterate through all combinations of alpha, gamma, epsilon values, algorithms, and number of episodes for each environment
for env_name in env_names:
    for alpha in alpha_range:
        for gamma in gamma_range:
            for epsilon in epsilon_range:
                for algorithm in algorithms:
                    for num_episodes in num_episodes_range:
                        # Create the environment
                        env = gym.make(env_name)

                        # Run your reinforcement learning algorithm here
                        total_reward = 0
                        if algorithm == 'Q-learning':
                            # Run Q-learning algorithm
                            agent = QLearning(
                                env.observation_space.n, env.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon
                            )

                            total_reward = train(env, agent, num_episodes)
                            agent.reset()

                        elif algorithm == 'DoubleQ-learning':
                            # Run DoubleQ-learning algorithm
                            agent = DoubleQLearning(
                                env.observation_space.n, env.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon
                            )

                            total_reward = train(env, agent, num_episodes)
                            agent.reset()
                            

                        # Add the results to the list
                        results.append((env_name, algorithm, alpha, gamma, epsilon, num_episodes, total_reward))

                        # Close the environment
                        env.close()

# Sort the results by total reward in descending order
results.sort(key=lambda x: x[6], reverse=True)

# Save the results to a CSV file
with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['env_name', 'algorithm', 'alpha', 'gamma', 'epsilon', 'num_episodes', 'total_reward'])
    writer.writerows(results)
   
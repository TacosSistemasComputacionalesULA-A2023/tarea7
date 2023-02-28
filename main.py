import csv
import multiprocessing
from utils import Arguments, run_training


if __name__ == '__main__':

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
    arguments = []
    for env_name in env_names:
        for alpha in alpha_range:
            for gamma in gamma_range:
                for epsilon in epsilon_range:
                    for algorithm in algorithms:
                        for num_episodes in num_episodes_range:
                            arguments.append(Arguments(
                                env_name,
                                algorithm,
                                num_episodes,
                                alpha,
                                gamma,
                                epsilon,
                            ))

    p = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    result = p.map_async(run_training, arguments)
    values = result.get()

    # Sort the results by total reward in descending order
    values.sort(key=lambda x: x[6], reverse=True)

    # Save the results to a CSV file
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['env_name', 'algorithm', 'alpha', 'gamma',
                        'epsilon', 'num_episodes', 'total_reward'])
        writer.writerows(values)

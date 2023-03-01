import csv
import multiprocessing
from utils import Arguments, run_training, init_data
import time
import datetime


if __name__ == '__main__':
    start = time.time()

    # Define the environment names
    env_names = ["CliffWalking-v0", "Taxi-v3"]

    # Define the range of values for alpha, gamma, and epsilon
    alpha_range = [0.5]
    gamma_range = [0.9]
    epsilon_range = [0.1]

    # Define the reinforcement learning algorithms
    algorithms = ['Q-learning', 'DoubleQ-learning']

    # Define the range of values for the number of episodes
    num_episodes_range = range(1, 500, 10)

    # Create a list to store the results
    results = []

    # Iterate through all combinations of alpha, gamma, epsilon values, algorithms, and number of episodes for each environment
    arguments = []
    for epsilon in epsilon_range:
        for alpha in alpha_range:
            for gamma in gamma_range:
                for env_name in env_names:
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

    counter = multiprocessing.Value('i', len(arguments))
    p = multiprocessing.Pool(initializer=init_data, initargs=(counter,len(arguments)), processes=multiprocessing.cpu_count())
    result = p.map_async(run_training, arguments)
    values = result.get()

    # Save the results to a CSV file
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['env_name', 'algorithm', 'alpha', 'gamma',
                        'epsilon', 'num_episodes', 'total_reward'])
        writer.writerows(values)

    print(f'Time taken: {datetime.timedelta(seconds=time.time() - start)}')
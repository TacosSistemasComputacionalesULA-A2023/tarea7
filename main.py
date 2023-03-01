import csv
import multiprocessing
from utils import Arguments, run_training, init_data, NUM_EPISODES
import time
import datetime


if __name__ == '__main__':
    start = time.time()

    # Define the environment names
    env_names = ["CliffWalking-v0", "Taxi-v3"]

    # Define the range of values for alpha, gamma, and epsilon
    alpha_range = [0.3, 0.6, 0.9]
    gamma_range = [0.3, 0.6, 0.9]
    epsilon_range = [0.3, 0.6, 0.9]

    # Define the reinforcement learning algorithms
    algorithms = ['Q-learning', 'DoubleQ-learning']

    # Create a list to store the results
    results = []

    # Iterate through all combinations of alpha, gamma, epsilon values, algorithms, and number of episodes for each environment
    arguments = []
    for epsilon in epsilon_range:
        for alpha in alpha_range:
            for gamma in gamma_range:
                for env_name in env_names:
                    for algorithm in algorithms:
                        arguments.append(Arguments(
                            env_name,
                            algorithm,
                            alpha,
                            gamma,
                            epsilon,
                        ))

    counter = multiprocessing.Value('i', len(arguments))
    p = multiprocessing.Pool(initializer=init_data, initargs=(
        counter, len(arguments)), processes=multiprocessing.cpu_count())
    result = p.map_async(run_training, arguments)
    values = result.get()

    # Save the results to a CSV file
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['env_name', 'algorithm', 'alpha', 'gamma',
                        'epsilon', 'num_episodes', 'total_reward', 'end_episode_reward', 'episode'])
        for value in values:
            for i, episode_reward in enumerate(value[-1]):
                writer.writerow([value[0], value[1], value[2], value[3], value[4], value[5], value[6], episode_reward, i])

    print(f'Time taken: {datetime.timedelta(seconds=time.time() - start)}')

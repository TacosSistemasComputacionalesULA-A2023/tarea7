import matplotlib.pyplot as plt
import pandas as pd
import csv

datos = pd.read_csv('results.csv')

# Seleccionamos las dos columnas que deseamos graficar
columnas = ['end_episode_reward', 'env_name', 'algorithm', 'alpha', 'gamma', 'epsilon', 'episode']

# Dividimos los datos en grupos de 500 líneas
grupos = [datos[columnas].iloc[i:i+500,:] for i in range(0, len(datos), 501)]

# Create a list to store the results
results = []

# Creamos una gráfica para cada grupo de datos
for i in range(0, len(grupos), 2):
    sample = grupos[i].values[0]
    
    # Graficamos los ingresos y los gastos en función del tiempo
    plt.plot(grupos[i]['episode'], grupos[i]['end_episode_reward'])
    plt.plot(grupos[i+1]['episode'], grupos[i+1]['end_episode_reward'])
    
    # Agregamos títulos y etiquetas a los ejes
    plt.title(f'Env: {sample[1]} w/ Q y 2Q A: {sample[3]} G: {sample[4]} E: {sample[5]}')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    
    # Agregamos una leyenda
    plt.legend(['Q-Learning', 'Double-QLearning'])

    # Mostramos la gráfica
    plt.show()

# Save the results to a CSV file
# with open('analysis.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
    
#     writer.writerow(['env_name', 'algorithm', 'alpha', 'gamma',
#                         'epsilon', 'minimum_reward', 'maximum_reward'])
     
#     for result in results:
#         writer.writerow(result)
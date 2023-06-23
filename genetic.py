"""
Título do Programa: AG e 2-opt para o TSP star100

Descrição:
Implementação de um algoritmo genético, juntamente com o 2-opt para a resolução
do problema do caixeiro viajante das 100 estrelas.

"""

# Imports 
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Hiperparametros
n_stars = 99
n_population = 700
n_interations = 200
mutation_rate = 1
stars = []
stars_list = np.arange(1,101)
points = []
index = 1


# função para ler o arquivo com as coordenadas das estrelas
def read_file():
    index = 1
    with open('star100.xyz') as file:
        for line in file.readlines():
            star = line.split(' ')
            # Armazenando as coordenadas das estrelas em um dicionário
            stars.append(dict(index=int(index), x=float(star[0]), y=float(star[1]), z=float(star[2])))
            points.append((float(star[0]), float(star[1]), float(star[2])))
            index += 1 

# Função para criar as populações
def genesis(n_population):
    population_set = []
    for _ in range(n_population):
        # Geração de um individuo
       # tour = [1] + list(np.random.choice(list(range(2, n_stars + 1)), n_stars - 1, replace=False))
        tour = random.sample(range(2, n_stars + 2), 99)
        population_set.append(tour)
    return np.array(population_set)

# função para calcular a distancia entre duas estrelas
def distance(star1: dict, star2: dict):
    return math.sqrt((star1['x'] - star2['x']) ** 2 + (star1['y'] - star2['y']) ** 2 + (star1['z'] - star2['z']) ** 2)

def dist(star_a, star_b, stars):
    return distance(stars[star_a], stars[star_b])

# Função para avaliar a aptidão (fitness) de uma solução
def fitness_eval(stars_list, stars):
    total = 0
    for i in range(n_stars-1):
        a = stars_list[i]-1
        b = stars_list[i+1]-1
        total += dist(a,b, stars)
    return total

# Obtém a aptidão (fitness) de todas as soluções na população
def get_all_fitnes(population_set, stars):
    fitnes_list = np.zeros(n_population)
    
    # Itera sobre todas as soluções, computando a aptidão de cada uma
    for i in  range(n_population):
        fitnes_list[i] = fitness_eval(population_set[i], stars)

    return fitnes_list

# Seleção dos progenitores para reprodução
def progenitor_selection(population_set,fitnes_list):
    total_fit = fitnes_list.sum()
    prob_list = fitnes_list/total_fit
    
    # Observe que há a chance de um progenitor se reproduzir com ele mesmo
    progenitor_list_a = np.random.choice(list(range(len(population_set))), len(population_set),p=prob_list, replace=True)
    progenitor_list_b = np.random.choice(list(range(len(population_set))), len(population_set),p=prob_list, replace=True)
    
    progenitor_list_a = population_set[progenitor_list_a]
    progenitor_list_b = population_set[progenitor_list_b]
    
    mutation_rate
    return np.array([progenitor_list_a,progenitor_list_b])

# Cruzamento dos progenitores para gerar a nova população
def mate_progenitors(prog_a, prog_b):
    offspring = prog_a[0:5]

    for city in prog_b:

        if not city in offspring:
            offspring = np.concatenate((offspring,[city]))

    return offspring
                
# Mutação de uma solução
def mate_population(progenitor_list):
    new_population_set = []
    for i in range(progenitor_list.shape[1]):
        prog_a, prog_b = progenitor_list[0][i], progenitor_list[1][i]
        offspring = mate_progenitors(prog_a, prog_b)
        new_population_set.append(offspring)
        
    return new_population_set

# Mutação de uma solução
def mutate_offspring(offspring):
    for q in range(int(n_stars*mutation_rate)):
        a = np.random.randint(0,n_stars)
        b = np.random.randint(0,n_stars)

        offspring[a], offspring[b] = offspring[b], offspring[a]

    return offspring
       
def mutate_population(new_population_set):
    mutated_pop = []
    for offspring in new_population_set:
        mutated_pop.append(mutate_offspring(offspring))
    return mutated_pop

# Função para auxiliar o cálculo da distância percorrida por uma rota para o 2-opt
def calculate_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        distance += dist(route[i]-1,route[i + 1]-1,stars)
    return distance

# Algoritmo de melhoria 2-opt
def two_opt(route):
    improved = True
    best_route = route
    best_distance = calculate_distance(route)
    
    while improved:
        improved = False
        # Precisamos garantir que ele não retire o sol da primeira e última posição
        for i in range(1, len(route) - 3): # Ajuste do limite superior do primeiro loop
            for j in range(i + 2, len(route)-1): # Ajuste do limite inferior e superior do segundo loop
                new_route = route.copy()
                new_route[i:j+1] = route[j:i-1:-1]

                new_distance = calculate_distance(new_route)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
        route = best_route

    return best_route, best_distance
  
# Chamada das funções para executar o algoritmo genético
read_file()
population_set = genesis(n_population)
fitnes_list = get_all_fitnes(population_set,stars)
progenitor_list = progenitor_selection(population_set,fitnes_list)
new_population_set = mate_population(progenitor_list)
print(new_population_set[0])
mutated_pop = mutate_population(new_population_set)
print(mutated_pop[0])
mutated_pop[0]
best_solution = [-1,np.inf,np.array([])]
finder = [np.inf,np.array([])]
fitnes_mean = [np.inf]

# Loop para realizar cada interação do algoritmo genético
for i in range(n_interations):
    if i%100==0: print(i, fitnes_list.min(), fitnes_list.mean())
    fitnes_list = get_all_fitnes(mutated_pop,stars)
    
    fitnes_mean.append(fitnes_list.mean())
    # pegamos a solução com o menor valor de fitness
    selected_rows = np.array(mutated_pop)[fitnes_list.min() == fitnes_list]
    # caso haja mais de uma solução com o mesmo valor de fitness, escolhe uma delas aleatoriamente
    selected_index = np.random.choice(len(selected_rows))
    finder[0] = fitnes_list.min()

    # Acrescentamos o sol na primeira e última posição e atualizamos o valor do fitness
    finder[1] = np.insert(selected_rows[selected_index], 0, 1)
    finder[1] = np.append(finder[1], 1)
    finder[0] += dist(finder[1][0]-1,finder[1][1]-1,stars)
    finder[0] += dist(finder[1][99]-1,finder[1][100]-1,stars) 

    # Para salvar a melhor solução encontrada
    if finder[0] < best_solution[1]:
        best_solution[0] = i
        best_solution[1] = finder[0]
        best_solution[2] = finder[1]


    progenitor_list = progenitor_selection(population_set,fitnes_list)
    new_population_set = mate_population(progenitor_list)
    
    mutated_pop = mutate_population(new_population_set)

print(f"teste {best_solution}")
best_solution[2], best_solution[1] = two_opt(best_solution[2])
print(f"Melhor Solução: {best_solution}")

# Plot da média dos valores de fitness
plt.plot(range(n_interations+1), fitnes_mean)
plt.xlabel('Iterações')
plt.ylabel('Média do Fitness')
plt.title('Evolução da Média do Fitness')
plt.show()
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
import time
from plot import *
from copy import *

# Hiperparametros
n_stars = 99
n_population = 500
n_interations = 1000
mutation_rate = 0.01

# Variáveis
stars = []
stars_list = np.arange(1,101)
points = []
index = 1


# Função para ler o arquivo com as coordenadas das estrelas
def read_file():
    index = 1
    with open('star100.xyz') as file:
        for line in file.readlines():
            star = line.split(' ')
            # Armazenando as coordenadas das estrelas em um dicionário
            stars.append(dict(index=int(index), x=float(star[0]), y=float(star[1]), z=float(star[2])))
            points.append((float(star[0]), float(star[1]), float(star[2])))
            index += 1 

# Função para criar a população inicial
def genesis(n_population):
    population_set = []
    for _ in range(n_population):
        # Geração de um individuo
        tour = random.sample(range(2, n_stars + 2), 99)
        population_set.append(tour)
    return np.array(population_set)

# Função para calcular a distancia entre duas estrelas
def distance(star1: dict, star2: dict):
    return math.sqrt((star1['x'] - star2['x']) ** 2 + (star1['y'] - star2['y']) ** 2 + (star1['z'] - star2['z']) ** 2)

def dist(star_a, star_b, stars):
    return distance(stars[int(star_a)], stars[int(star_b)])

# Função para avaliar a aptidão (fitness) de uma solução
def fitness_eval(stars_list, stars):
    total = 0
    for i in range(n_stars-1):
        a = stars_list[i]-1
        b = stars_list[i+1]-1
        total += dist(a,b, stars)
    return total

# Obtém a aptidão (fitness) de todas as soluções na população
def get_all_fitnes(population_set, stars, size):
    fitnes_list = np.zeros(size)
    
    # Itera sobre todas as soluções, computando a aptidão de cada uma
    for i in  range(size):
        fitnes_list[i] = fitness_eval(population_set[i], stars)

    return fitnes_list

def selectParents(population,tournamentSize=3):
    parents = []
    fitnessValues = np.zeros(tournamentSize)

    # Torneio 
    for _ in range(2):
        candidates = random.choices(population, k=tournamentSize) 
        fitnessValues = get_all_fitnes(candidates,stars,tournamentSize)
        index = np.argmin(fitnessValues)
        # pegamos a solução com o menor valor de fitness
        selected_rows = candidates[index]
       
        parents.append(selected_rows)
    
    return parents[0], parents[1]

def generateChildren(parent1, parent2):
    # Dois pontos de crossover para garantir melhor variabilidade
    crossoverPont1 = random.randint(0, len(parent1)-1)
    crossoverPont2 = random.randint(0, len(parent1)-1)
    if crossoverPont2 < crossoverPont1:
        crossoverPont1, crossoverPont2 = crossoverPont2, crossoverPont1

    # Geração de dois filhos

    part1 = parent1[0:crossoverPont1]
    aux = crossoverPont1
    # Verificação para garantir que haja apenas uma estrela em cada indivíduo
    part2 = []
    for i in range(crossoverPont1,crossoverPont2):
        if not parent2[i] in part1:
            part2.append(parent2[i])
    
    child1 = np.concatenate((part1, part2))
    part3 = []
    for point in parent1:
        aux += 1
        if not point in child1:
            part3.append(point)
        
                
    child1 = np.concatenate((child1,part3))

    part1 = parent2[0:crossoverPont1]
    aux = crossoverPont1
    
    part2 = []
    for i in range(crossoverPont1,crossoverPont2):
        if not parent1[i] in part1:
            part2.append(parent1[i])
    
    child2 = np.concatenate((part1, part2))
    part3 = []
    for point in parent2:
        aux += 1
        if not point in child2:
            part3.append(point)
        
                
    child2 = np.concatenate((child2,part3))

    return child1, child2

def mutation(individual):
    mutatedIndividual = individual.copy()

    for _ in range(len(mutatedIndividual)):
        if random.random() < mutation_rate:
            a = np.random.randint(0,n_stars)
            b = np.random.randint(0,n_stars)
            mutatedIndividual[a], mutatedIndividual[b] = mutatedIndividual[b], mutatedIndividual[a]

    return mutatedIndividual 

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
  
# Execução
# Chamada das funções para executar o algoritmo genético
read_file()
beginAG = time.time() # inicio da contagem de tempo do AG
population_set = genesis(n_population) 
fitnes_list = get_all_fitnes(population_set,stars,n_population)
new_population = population_set

best_solution = [-1,np.inf,np.array([])]
finder = [np.inf,np.array([])]
fitnes_mean = [np.inf]
fitnes_min = [np.inf]
random.seed(4)

# Loop para realizar cada interação do algoritmo genético
for i in range(n_interations):
    if i%100==0: print(i, fitnes_list.min(), fitnes_list.mean())
    fitnes_list = get_all_fitnes(new_population,stars,n_population)
    
    fitnes_mean.append(fitnes_list.mean())
    fitnes_min.append(fitnes_list.min())
    # pegamos a solução com o menor valor de fitness
    selected_rows = np.array(new_population)[fitnes_list.min() == fitnes_list]
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

    aux = new_population
    new_population = np.zeros((0,99))
   
    # Loop com metade do tamanho da população, pois temos dois filhos
    for _ in range(n_population//2):
        father1, father2 = selectParents(aux)

        Child1, Child2 = generateChildren(father1, father2)
        mutateChild1 = mutation(Child1)
        mutateChild2 = mutation(Child2)
        new_population = np.vstack((new_population,mutateChild1))
        new_population = np.vstack((new_population,mutateChild2))
    
endAG = time.time() # fim da contagem de tempo do OPT

print(f"Melhor solução AG: {best_solution}")
#plot_graphic_mean(n_interations,fitnes_mean)
#plot_graphic_min(n_interations,fitnes_min)
#plot_fork(stars,best_solution[2])

beginOPT = time.time() # inicio da contagem de tempo do OPT
# Chamada do algoritmo de otimização 2-opt
best_solution[2], best_solution[1] = two_opt(best_solution[2])
endOPT = time.time() # fim da contagem de tempo do OPT
print(f"Solução otimizada 2-opt: {best_solution}")


#plot_fork(stars,best_solution[2])
print("Tempo gasto pelo AG: ",endAG - beginAG)
print("Tempo gasto pelo OPT: ",endOPT - beginOPT)

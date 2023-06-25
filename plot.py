import matplotlib.pyplot as plt
import numpy  as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def plot_graphic_min(n_interations,list):
    # Plot da média dos valores de fitness
    plt.plot(range(n_interations+1), list, color = 'red')
    plt.grid(True,color =  "#d0d1d2")
    plt.xlabel('Iterações')
    plt.ylabel('Valor do Fitness')
    plt.title('Menor valor do Fitness de cada população')
    plt.savefig('imagens/10-min.pdf', format='pdf')
    plt.show()

def plot_graphic_mean(n_interations,list):
    # Plot da média dos valores de fitness
    plt.plot(range(n_interations+1), list, color = 'green')
    plt.grid(True,color =  "#d0d1d2")
    plt.xlabel('Iterações')
    plt.ylabel('Média do Fitness')
    plt.title('Evolução da Média do Fitness')
    plt.savefig('imagens/10-mean.pdf', format='pdf')
    plt.show()

def plot_fork(points, path: list):
    xyz = [[0 for _ in range(3)] for _ in range(101)]

    X = []
    Y = []
    Z = []
    for i in range(0, len(path)):
        if (i < len(path)):
            xyz[i][0] = points[int(path[i]-1)]['x']
            xyz[i][1] = points[int(path[i])-1]['y']
            xyz[i][2] = points[int(path[i])-1]['z']

    ax = plt.axes(projection='3d')
    X= [sub[0] for sub in xyz]
    Y= [sub[1] for sub in xyz]
    Z= [sub[2] for sub in xyz]
    ax.plot(X, Y, Z, color='gray', alpha=0.3)
    X.pop(X.index(0.0))
    Y.pop(Y.index(0.0))
    Z.pop(Z.index(0.0))
    ax.scatter(0, 0, 0, s=200, color='yellow')
    ax.scatter(X, Y, Z, color='black')
    plt.savefig('imagens/10-fork.pdf', format='pdf')
    plt.show()

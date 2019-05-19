
import re
import sys
from ast import literal_eval
import numpy as np
import os
import random
import matplotlib.pyplot as plt


num_of_cities = None
RECHARGINGSTATIONS = None
DISTANCE_MATRIX = None
BAT_LIFE_UAVS = None
######################################################################## Generate  ###############################################################################################################

def genPathVector(size):
    stations = list(range(1, size))
    random.shuffle(stations)
    stations.insert(0, 0)
    val = tuple(stations)
    #print(val)
    return val

######################################################################### read and write values - Save / Load ####################################################################################

# read NO_OF_UAVS from json or init file according to responseMain
def getBAT_LIFE_UAVS():
    BAT_LIFE_UAVS = open('InstanceData.txt', "r").readlines()[2]
    return int(BAT_LIFE_UAVS)

def readRechargingStationsVector():
    RECHARGINGSTATIONS = open('InstanceData.txt', "r").readlines()[4]
    RECHARGINGSTATIONS = literal_eval(RECHARGINGSTATIONS)
    # results = [list(map(int, x)) for x in RECHARGINGSTATIONS]
    return RECHARGINGSTATIONS

def readnum_of_cities():
    num_of_cities = open('InstanceData.txt', "r").readlines()[0]
    return int(num_of_cities)


# open and read from txt file
def readDistanceMatrixTxt():
   fileVal = open('InstanceData.txt', "r")
   with fileVal as file:
        data= file.readlines()[5:(5 + num_of_cities)]
        # array = file.readlines()
        testsite_array = []
        for line in data:
            # val = line.strip().replace("   ", ",")
            val = re.sub("\s+", ",", line.strip())
            val.replace("\n", "")
            arr = val.split(',')
            testsite_array.append(arr)
        results = [list(map(int, x)) for x in testsite_array]
   return results

############################################################################### Save Data  #############################################################################

def saveInitDatantoFinalFile(filename,data):
        sval = str(data)
        filename = filename

        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        highscore = open(filename, append_write)
        highscore.write(sval + '\n')
        highscore.close()

################################################################################# population_generation ########################################################################
def generate_population(num_of_cities, population_size):
    population = []
    for ind in range(0,population_size):
        p=genPathVector(num_of_cities)
        population.append(p)
    return population
################################################################################  Evaluation  #######################################################################################

# block to find max value for penalisation
def findmax(M):
    I = len(M)
    J = len(M[0])
    # Initializing max element as INT_MIN
    maxElement = -sys.maxsize - 1
    # checking each element of matrix
    # if it is greater than maxElement,
    # update maxElement
    for i in range(I):
        for j in range(J):
            if (M[i][j] > maxElement):
                maxElement = M[i][j]
    # finally return maxElement
    finalVal = maxElement * (len(M[0]))
    return finalVal


def evaluate_solution(path_vector):
    pos = 0
    #current_city = path_vector[pos]
    M =  readDistanceMatrixTxt()
    nxt_city = path_vector[pos + 1]
    Kilometer = 0  # Distance variable
    Battery = 0
    p = findmax(DISTANCE_MATRIX)  # penalization p
    for i in (path_vector):
        current_city = path_vector[pos]
        if (RECHARGINGSTATIONS[current_city] == 1):
            Battery = Battery + BAT_LIFE_UAVS
        if (Battery >= M[current_city][nxt_city]):
            Kilometer = Kilometer + M[current_city][nxt_city]
            Battery = Battery - M[current_city][nxt_city]
        if (Battery < M[current_city][nxt_city]):
            #Kilometer = Kilometer + p
            Kilometer = Kilometer + p +1000
            break
        pos = pos + 1
        if (pos <= (len(path_vector) - 2)):
            nxt_city = path_vector[pos + 1]
        if (pos == (len(path_vector) - 1)):
            nxt_city = path_vector[0]

    return (Kilometer)

def evaluate_fitnessWithBattery1(Pop):
    fit=[]
    for i in range(0,(len(Pop))):
     path_vector = Pop[i]
     #print(len(Pop))
     fitness= evaluate_solution(path_vector)
     fit.append(fitness)
     #print(fit)
    return fit

def evaluate_fitnessWithBattery():
    path_vector = genPathVector(num_of_cities)
    dist = evaluate_solution(path_vector)
    return dist
##################################################################################  Random Search  ##################################################################################
def randomsearch():
    best_vector = genPathVector(num_of_cities)
    best_DIST = evaluate_solution(best_vector)

    for i in range(10000):
        new_vector = genPathVector(num_of_cities)
        new_DIST = evaluate_solution(new_vector)
        if new_DIST < best_DIST     :
            best_DIST = new_DIST
            best_vector = new_vector
    saveInitDatantoFinalFile("Result_RS_fitness"+".txt",best_DIST )
    saveInitDatantoFinalFile("Result_RS_Vector" + ".txt", best_vector)
    print("Best Distance for Random Search (Best Fitness)", best_DIST)

################################################################################## Evaluationary Algorithm-1-selction  ################################################################
def select_Population(population,scores):
    selectedPop = []
    #print(population)
    #print(scores)
    for ind in range(0, len(population)):
        ind = select_individual_by_tournament(population, scores)
        selectedPop.append(ind)
    return selectedPop

def select_individual_by_tournament(population,scores):
        population_size = len(scores)

        fighter_1 = random.randint(0, population_size - 1)
        fighter_2 = random.randint(0, population_size - 1)

        fighter_1_fitness = scores[fighter_1]
        fighter_2_fitness = scores[fighter_2]

        if fighter_1_fitness <= fighter_2_fitness:
            winner = fighter_1
        else:
            winner = fighter_2
        return population[winner]
################################################################################2-crossover################################################################################
def ordered_crossover(parent_1, parent_2):
        points_num = len(parent_1)
        cut_ix = np.random.choice(points_num - 2, 2, replace=False)
        min_ix = np.min(cut_ix)
        max_ix = np.max(cut_ix)
        offspring_1 = np.zeros(points_num)
        current_ix = 0
        set_1 = parent_1[min_ix:max_ix]
        for i, elem in enumerate(parent_2):
            if elem not in set_1:
                if current_ix != min_ix:
                    offspring_1[current_ix] = elem
                else:
                    current_ix = max_ix
                    offspring_1[current_ix] = elem
                current_ix += 1
        offspring_1[min_ix:max_ix] = set_1
        offspring_2 = np.zeros(points_num)
        current_ix = 0
        set_2 = parent_2[min_ix:max_ix]
        for i, elem in enumerate(parent_1):
            if elem not in set_2:
                if current_ix != min_ix:
                    offspring_2[current_ix] = elem
                else:
                    current_ix = max_ix
                    offspring_2[current_ix] = elem
                current_ix += 1
        offspring_2[min_ix:max_ix] = set_2
        return [int(i) for i in offspring_1], [int(i) for i in offspring_2]

def crossover_population(population):
    cross_pop=[]
    population_size = len(population)
    for ind in range(0, len(population)):
        parent1=population[random.randint(0, population_size - 1)]
        parent2=population[random.randint(0, population_size - 1)]
        CrossInd1,CrossInd2= ordered_crossover(parent1,parent2)
        cross_pop.append(CrossInd1)
        cross_pop.append(CrossInd2)
    return cross_pop
#######################################################################3-mutation#################################################################################################
def swap_mutation(individ, mutation_intensity):
    for _ in range(int(len(individ) * mutation_intensity)):
        if random.random() > 0.5:
            ix = np.random.choice(len(individ), 2, replace=False)
            tmp = individ[ix[0]]
            individ[ix[0]] = individ[ix[1]]
            individ[ix[1]] = tmp
    return individ

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    for ind in range(0, len(population)):
        mutatedInd =swap_mutation(population[ind],mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop
##########################################################################4-replacement####################################################################################
def replacement(Q,P):
    P2=[]
    P1 = P+Q
    #print("p1",P1)
    P2.append(P1[1])
    for ind in range(1, (len(P1)-1)):
        for ind1 in range(ind+1, (len(P1))):
            if P1[ind] != P1[ind1]:
               P2.append(P1[ind1])
               ind1+=1
        ind+=1
    return P1

def replacement1(Q,P):
    P = np.array(Q)
    return P


###############################################################################5-replacement##############################################################################
def ga_method():
  mutation_probability = 0.1
  population_size = 20
  no_of_evaluation=0
  best_fit_progress=[]
  Q = []
  P = generate_population(num_of_cities, population_size)
  scores = evaluate_fitnessWithBattery1(P)
  best_fit = findmax(readDistanceMatrixTxt())
  best_ind = []
  while no_of_evaluation < 12:
      Q = select_Population(P, scores)
      Q = crossover_population(Q)
      Q = mutatePopulation(Q, mutation_probability)
      P = replacement1(Q, P)
      for i in P:
          new_fit = evaluate_solution(i)
          if new_fit < best_fit:
              best_fit = new_fit
              best_ind = i
      #print("till now best",best_fit)
      # Plot progress
      best_fit_progress.append(best_fit)
      plt.plot(best_fit_progress)
      plt.xlabel('number of Evalution')
      plt.ylabel('Best fitness (% target)')
      no_of_evaluation+=1
  print("Best fitness for Genetic Algorithm", best_fit)
  return best_fit, best_ind


def GAalgorithm():
    fitness,vector = ga_method()
    saveInitDatantoFinalFile("Result_EA_Fitness"+".txt",fitness)
    saveInitDatantoFinalFile("Result_EA_vector" + ".txt", vector)
    plt.show()
    #print("Best fitness for Genetic Algorithm", fitness)
    return fitness
##################################################################################  Main  ##################################################################################
num_of_cities = readnum_of_cities()
RECHARGINGSTATIONS = readRechargingStationsVector()
DISTANCE_MATRIX = readDistanceMatrixTxt()
BAT_LIFE_UAVS = getBAT_LIFE_UAVS()

def main():

    #####random search
    randomsearch()

    #####Genetic algo
    GAalgorithm()

if __name__ == '__main__':
    main()

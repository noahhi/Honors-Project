import numpy as np
from numpy.random import randint


class CMDKP:
  '''
  Used to represent an instance of the Cubic Multidimensional Knapsack Problem

  Attributes:
    n (int): problem size (i.e., the number of variables)
    density (int): the density of the value coefficients
    constraints (int): the number of constraints
  '''

  def __init__(self, n, density, constraints, seed=0):
    np.random.seed(seed)

    self.n = n
    self.density = density
    self.constraints = constraints

    self.c = np.zeros(n)
    self.C = np.zeros((n,n))
    self.D = np.zeros((n,n,n))

    for i in range(n):
      if randint(1, 101) <= density:
        self.c[i] = randint(low=1,high=101)
      for j in range(i+1, n):
        if randint(1, 101) <= density:
          self.C[i,j] = randint(low=1,high=101)
        for k in range(j+1, n):
          if randint(1, 101) <= density:
            self.D[i,j,k] = randint(low=1,high=101)

    # generate item weights (each item has a weight in each dimension)
    self.a = np.random.randint(low=1, high=51, size=(n,constraints))

    # generate knapsack capacities (one for each dimension)
    self.b = np.zeros(constraints)
    for j in range(constraints):
      self.b[j] = np.random.randint(low=50, high=sum(self.a[i,j] for i in range(n))) #TODO what should lower bound be? 50 -> errors for small n


def main():
    cdmkp = CMDKP(n=4,density=80,constraints=2,seed=1)


if __name__=="__main__":
  main()

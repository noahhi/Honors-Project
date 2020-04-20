'''
File for generating instances of Cubic problems
'''

import numpy as np
from numpy.random import randint


class Cubic:
  '''
  Base class for all Cubics

  Attributes:
  	n (int): problem size/number of decision variables
  	density (int): density of value coefficients (should be in range [0,100]
  	c ([int]): linear values
  	C ([[int]]): quadratic (pair) values
  	D ([[[int]]]): cubic (triplet) values
  '''

  def __init__(self, n, density, seed=0):
    np.random.seed(seed)

    self.n = n
    self.density = density

    self.c = np.zeros(n)
    self.C = np.zeros((n,n))
    self.D = np.zeros((n,n,n))

    # randomly initialize all value coefficients
    for i in range(n):
      if randint(1, 101) <= density:
        self.c[i] = randint(low=1,high=101)
      for j in range(i+1, n):
        if randint(1, 101) <= density:
          self.C[i,j] = randint(low=1,high=101)
        for k in range(j+1, n):
          if randint(1, 101) <= density:
            self.D[i,j,k] = randint(low=1,high=101)


class CMDKP(Cubic):
  '''
  Used to represent an instance of the Cubic Multidimensional Knapsack Problem. Subclass of Cubic.

  Additional Attributes:
    constraints (int): the number of constraints (i.e. dimensions)
  '''

  def __init__(self, n, density, constraints, seed=0):
    super().__init__(n=n, density=density, seed=seed)

    self.constraints = constraints

    # generate item weights (each item has a weight in each dimension)
    self.a = np.random.randint(low=1, high=51, size=(n,constraints))

    # generate knapsack capacities (one for each dimension)
    self.b = np.zeros(constraints)
    for j in range(constraints):
      self.b[j] = np.random.randint(low=50, high=sum(self.a[i,j] for i in range(n))) #TODO what should lower bound be? 50 -> errors for small n


class CMKP(Cubic):
  '''
  Used to repesent an instance of the Cubic Multiple Knapsack Problem

  Additional Attributes:
    m (int): the number of knapsacks
  '''

  def __init__(self, n, density, m, seed=0):
    super().__init__(n=n, density=density, seed=seed)

    self.m = m

    # generate item weights
    self.a = np.random.randint(low=1, high=51, size=n)
    total_weight = sum(self.a)

    # give all knaps the same initial capacity
    knap_cap = total_weight*0.8/m

    # generate knapsack capacities (one for each knapsack)
    self.b = np.zeros(m)
    for j in range(m):
      # self.b[j] = np.random.randint(low=50, high=total_weight)
      self.b[j] = knap_cap


def main():
    cdmkp = CMDKP(n=50,density=60,constraints=3)
    cmkp = CMKP(n=50,density=60,m=3)


if __name__=="__main__":
  main()

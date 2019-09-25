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

    self.P1 = np.zeros(n)
    self.P2 = np.zeros((n,n))
    self.P3 = np.zeros((n,n,n))

    for i in range(n):
      if np.random.randint(1, 101) <= density:
        self.P1[i] = np.random.randint(low=1,high=101)
      for j in range(i+1, n):
        if np.random.randint(1, 101) <= density:
          self.P2[i,j] = np.random.randint(low=1,high=101)
        for k in range(j+1, n):
          if np.random.randint(1, 101) <= density:
            self.P3[i,j,k] = np.random.randint(low=1,high=101)

    # # generate linear, quadratic, and cubic profit matrices
    # self.P1 = np.random.randint(low=1, high=101, size=n)
    # self.P2 = np.random.randint(low=1, high=101, size=(n,n))
    # np.fill_diagonal(self.P2,0)
    # self.P3 = np.random.randint(low=1, high=101, size=(n,n,n))
    # np.fill_diagonal(self.P3,0)
    #
    # # helper function to replace values with 0s according to density
    # def add_zeros(x):
    #   if np.random.randint(low=1,high=101) > density:
    #     return 0
    #   else:
    #     return x
    # vectorized_zero = np.vectorize(add_zeros)
    #
    # self.P1 = vectorized_zero(self.P1)
    # self.P2 = vectorized_zero(self.P2)
    # self.P3 = vectorized_zero(self.P3)

    # generate item weights (each item has a weight in each dimension)
    self.w = np.random.randint(low=1, high=51, size=(n,constraints))

    # generate knapsack capacities (one for each dimension)
    self.C = np.zeros(constraints)
    for j in range(constraints):
      self.C[j] = np.random.randint(low=50, high=sum(self.w[i,j] for i in range(n))) #TODO what should lower bound be? 50 -> errors for small n


def main():
    cdmkp = CMDKP(n=4,density=80,constraints=2,seed=1)
    print(cdmkp.P1)
    print()
    print(cdmkp.P2)
    print()
    print(cdmkp.P3)


if __name__=="__main__":
  main()

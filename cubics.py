import numpy as np

'''
Used to represent an instance of the Cubic Multidimensional Knapsack Problem
'''
class CMDKP:
  def __init__(self, n, dimensions):
    self.n = n
    self.dimensions = dimensions

    # generate linear, quadratic, and cubic profit matrices
    self.P1 = np.random.randint(low=1, high=101, size=n)
    self.P2 = np.random.randint(low=1, high=101, size=(n,n))
    self.P3 = np.random.randint(low=1, high=101, size=(n,n,n))

    # generate item weights (each item has a weight in each dimension)
    self.w = np.random.randint(low=1, high=51, size=(n,dimensions))

    # generate knapsack capacities (one for each dimension)
    self.C = np.random.randint(low=50, high=1001, size=dimensions) # TODO what should upper bound for capacity be?

def main():
    cdmkp = CMDKP(5,1)

if __name__=="__main__":
  main()

from cubics import *
from utility import *
import numpy as np
import math

def naive_greedy(cubic, knap_selection_option=1, verbose=True):
    n = cubic.n
    c = cubic.c
    C = cubic.C
    D = cubic.D
    a = cubic.a
    b = cubic.b
    m = cubic.m

    print(b)

    # index set for variables x_i currently set to 1
    # 1 set for each knapsack to store items in that knapsack
    N1 = dict()
    for k in range(m):
        N1[k] = set()

    # set of knapsacks which still have room for more items
    activeK = set()
    for k in range(m):
        activeK.add(k)

    # init room remaining in each knapsack
    RHS = np.zeros(m)
    for i in range(m):
        RHS[i] = b[i]

    # initialize NK for each knapsack with all item indices whose weight < capacity
    # Nk[k] is the set of indices which can still fit in knapsack k (analagous to N0)
    N0 = {}
    for k in activeK:
        N0[k] = set()
        for i in range(n):
            if a[i] < RHS[k]:
                N0[k].add(i)

    while len(activeK) > 0:
        # apply knapsack choice selection rule for selecting knapsack k*
        knapsack_choice = -1

        # choose knapsack with lowest remaining capacity
        if knap_selection_option == 1:
            minRHS = math.inf
            for k in activeK:
                if RHS[k] < minRHS:
                    minRHS = RHS[k]
                    knapsack_choice = k

        # choose knapsack with the largest remaining capacity
        elif knap_selection_option == 2:
            maxRHS = -math.inf
            for k in activeK:
                if RHS[k] > maxRHS:
                    maxRHS = RHS[k]
                    knapsack_choice = k


        if verbose:
            print(f"k* by option {knap_selection_option} is {knapsack_choice}")
            print()


        # apply combination evaluation method to knpasack k*
        best_item = get_item_naive(cubic, N0=N0[k], N1=N1[k])

        # update N1 and RHS for choosen knapsack
        N1[knapsack_choice].add(best_item)
        RHS[knapsack_choice] -= a[best_item]

        # update N0 (remove items which would put us over capacity if taken)
        for i in N0[knapsack_choice].copy():
            if a[i] > RHS[knapsack_choice]:
                N0[knapsack_choice].remove(i)

        # remove item from N0 pool for all knapsacks
        for k in activeK.copy():
            if best_item in N0[k]:
                N0[k].remove(best_item)

            # remove knapsack from activeK if no more items can be fit
            if len(N0[k]) == 0:
                activeK.remove(k)

    return N1


'''
Retrieve and return the highest value item given a single knapsack.

Params:
    N0 : set of items to be considered
    N1 : set of items already in knapsack
'''
def get_item_naive(cubic, N0, N1):
    c = cubic.c
    C = cubic.C
    D = cubic.D
    a = cubic.a

    N1 = list(N1)

    value_ratios = {}
    for i in N0:
        value_ratios[i] = c[i] / a[i]

        for j in range(len(N1)):
            value_ratios[i] += C[i,N1[j]] + C[N1[j],i]
            for k in range(j+1, len(N1)):
                value_ratios[i] += getDValue(i,N1[j],N1[k],D)

    return max(value_ratios, key=lambda key:value_ratios[key])


def main():
    n = 10
    cmkp = CMKP(n=n, density=70, m=3)
    indices = naive_greedy(cmkp)
    print(indices)


if __name__=="__main__":
  main()

from cubics import *

'''
This function returns the value of the D_ijk coefficient by putting the i, j, k in order
'''
def getDValue(i, j, k, D):

    # Put i, j, k in order
    if (i > j):
        midIndex = i
        lowIndex = j
    else:
        midIndex = j
        lowIndex = i

    if (midIndex > k):
        highIndex = midIndex
        if (lowIndex > k):
            midIndex = lowIndex
            lowIndex = k
        else:
            midIndex = k
    else:
        highIndex = k

    # Return the value of D
    return D[lowIndex, midIndex, highIndex];


'''
Given a set of variable indices, and the objective coefficients, compute the objective function value.
(Assume solution is feasible)
'''
def getObjVal(indices, c, C, D):
    objVal = 0

    # add up linear contributions
    for index in indices:
        objVal += c[index]

    # add up quadratic contributions
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            objVal += C[i,j]

            # add up cubic contributions
            for k in range(j+1, len(indices)):
                objVal += D[i,j,k]

    return objVal


'''
Apply Glover's constructive Greedy heuristic to a single dimension cubic knapsack
'''
def greedy(cubic):
    n = cubic.n
    c = cubic.c
    C = cubic.C
    D = cubic.D
    a = cubic.a
    b = cubic.b

    # index set for variables x_i currently set to 1
    N1 = set()
    # index set for variables x_i that can feasbily fit
    N0 = set()
    # room left in knapsack
    RHS = b[0]
    # initialize N0 with all item indices whose weight < capacity
    for i in range(n):
        if a[i] < RHS:
            N0.add(i)

    # compute intial value ratios
    # NOTE: initially only the linear contributions are considered
    value_ratios = np.zeros(n)
    for i in N0:
        value_ratios[i] = c[i] / a[i][0]

    # repeatedly pick the best item until no more items can fit
    while RHS > 0 and len(N0) > 0:
        # get the highest value item to include next
        take_index = np.argmax(value_ratios)

        # remove this item from pool of unassigned items
        N0.remove(take_index)
        value_ratios[take_index] = -1

        # update remaining capacity
        RHS = RHS - a[take_index][0]

        # update N0 (remove items which would put us over capacity if taken)
        for i in N0.copy():
            if a[i][0] > RHS:
                N0.remove(i)

        # update value ratios
        for i in N0:
            # add in new quadratic contributions
            value_ratios[i] += (C[i, take_index] + C[take_index, i])/a[i][0]

        # add in new cubic contributions
        for j in N1:
            value_ratios[i] += getDValue(i, j, take_index, D)/a[i][0]

        # add item to taken items
        N1.add(take_index)

    # return selected indices
    return N1


n = 15
cdmkp = CMDKP(n=n, density=70, constraints=1)
indices = greedy(cdmkp)
val = getObjVal(indices, cdmkp.c, cdmkp.C, cdmkp.D)
print(indices)
print(val)

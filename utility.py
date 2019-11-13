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

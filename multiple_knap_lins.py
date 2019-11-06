'''
Use linearizations to solve instances of the cubic Multiple knapsack to optimality
'''

from cubics import *
from gurobipy import *


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
Apply the standard linearization to an instance of the cubic multiple knapsack and return the model
'''
def standard_lin(cubic):
    n = cubic.n # num of decisions variables (items)
    m = cubic.m # num of knapsacks (if m=1, this is just a regular cubic knapsack problem)
    c = cubic.c # linear coefficient values (n x 1)
    C = cubic.C # quadratic coeffiient values (n x n)
    D = cubic.D # cubic coefficient values (n x n x n)
    a = cubic.a # item weights (n x 1)
    b = cubic.b # knapsack capacities (m x 1)

    model = Model()

    x = model.addVars(n,m, name="binary_var", vtype=GRB.BINARY)
    y = model.addVars(n,n,m, name="quad_var", vtype=GRB.CONTINUOUS)
    z = model.addVars(n,n,n,m, name="cubic_var", vtype=GRB.CONTINUOUS)

    # ensure each knapsack is under capacity
    for r in range(m):
        model.addLConstr(quicksum(x[i,r]*a[i] for i in range(n)) <= b[r])

    # ensure that y_ijr = x_ir*x_jr and z_ijkr = x_ir*x_jr*x_kr
    for r in range(m):
        for i in range(n):
            for j in range(i+1,n):
                model.addLConstr(y[i,j,r] <= x[i,r])
                model.addLConstr(y[i,j,r] <= x[j,r])
                model.addLConstr(y[i,j,r] >= x[i,r] + x[j,r] - 1)
                model.addLConstr(y[i,j,r] >= 0)
                for k in range(j+1,n):
                    model.addLConstr(z[i,j,k,r] <= x[i,r])
                    model.addLConstr(z[i,j,k,r] <= x[j,r])
                    model.addLConstr(z[i,j,k,r] <= x[k,r])
                    model.addLConstr(z[i,j,k,r] >= x[i,r] + x[j,r] + x[k,r] - 2)
                    model.addLConstr(z[i,j,k,r] >= 0)

    linear_values = 0
    quad_values = 0
    cubic_values = 0
    # add up total obj value across all knapsacks
    for r in range(m):
        linear_values = quicksum(c[i]*x[i,r] for i in range(n))
        quad_values = quicksum(C[i][j]*y[i,j,r] for i in range(n-1) for j in range(i+1,n))
        cubic_values = quicksum(D[i][j][k]*z[i,j,k,r] for i in range(n-2) for j in range(i+1,n-1) for k in range(j+1,n))

    model.setObjective(linear_values+quad_values+cubic_values, sense=GRB.MAXIMIZE)

    return model


'''
Apply the Adams and Forrester linearization to an instance of the cubic multiple knapsack and return the model
'''
def adams_and_forrester_lin(cubic):
    n = cubic.n # num of decisions variables (items)
    m = cubic.m # num of knapsacks (if m=1, this is just a regular cubic knapsack problem)
    c = cubic.c # linear coefficient values (n x 1)
    C = cubic.C # quadratic coeffiient values (n x n)
    D = cubic.D # cubic coefficient values (n x n x n)
    a = cubic.a # item weights (n x 1)
    b = cubic.b # knapsack capacities (m x 1)

    # create a new empty model
    mdl = Model()

    # compute L_hat and U_hat
    L_hat = np.zeros((n,n))
    U_hat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D_row = D[i][j]
            pos_indices = D_row > 0
            U_hat[i][j] = sum(D_row[pos_indices])
            neg_indices = D_row < 0
            L_hat[i][j] = sum(D_row[neg_indices])

    # compute L_bar and U_bar
    L_bar = np.zeros(n)
    U_bar = np.zeros(n)
    combined = C + U_hat
    for j in range(n):
        for i in range(n):
            U_bar[j] += max(combined[i][j], 0)
            L_bar[j] += min(combined[i][j], 0)

    # declare variables
    x = mdl.addVars(n,m, name="binary_var", vtype=GRB.BINARY)
    psi = mdl.addVars(n,m, vtype=GRB.CONTINUOUS)
    tau = mdl.addVars(n,n,m, vtype=GRB.CONTINUOUS)

    for r in range(m):
        mdl.addLConstr(quicksum(x[i,r]*a[i] for i in range(n)) <= b[r])

    #g = np.zeros((n,m))
    g = [[None] * m for i in range(n)]
    for r in range(m):
        for j in range(n):
            g[j][r] = quicksum((C[i][j]+C[j][i])*x[i,r] for i in range(n))

    #h = np.zeros((n,n,m))
    h = [[[None] * m for i in range(n)] for j in range(n)]
    for r in range(m):
        for j in range(n):
            for i in range(n):
                h[i][j][r] = quicksum(getDValue(i,j,k,D)*x[k,r] for k in range(n))

    # g = [[None] * n for r in range(m)]
    # for r in range(m):
    #     for j in range(n):
    #         g[r][j] = quicksum(C[i][j]*x[i,r] for i in range(n) if i!=j)

    # h = [[[None] * n for i in range(n)] for r in range(m)]
    # for r in range(m):
    #     for j in range(n):
    #         for i in range(n):
    #             h[r][i][j] = quicksum(D[i][j][k]*x[k,r] for k in range(n) if k!=j)

    # add constraints
    for r in range(m):
        for j in range(n):
            mdl.addConstr(psi[j,r] >= L_bar[j]+(x[j,r]*(U_bar[j]-L_bar[j]))-g[j][r]-sum(U_hat[i][j]*x[i,r]-tau[i,j,r] for i in range(n) if i != j))
            mdl.addConstr(psi[j,r] >= 0)
            for i in range(n):
                if i == j:
                    continue
                mdl.addConstr(tau[i,j,r] >= L_hat[i][j]+(x[i,r]*(U_hat[i][j]-L_hat[i][j]))-h[i][j][r])
                mdl.addConstr(tau[i,j,r] >= 0)

    # set objective
    mdl.setObjective(quicksum(x[j,r]*(c[j]+U_bar[j])-psi[j,r] for j in range(n) for r in range(m)), sense=GRB.MAXIMIZE)

    return mdl


def main():
    setParam('OutputFlag',0)
    setParam('LogFile',"")

    # generate instance of CMKP
    n = 15
    cdmkp = CMKP(n=n,density=70,m=3)

    # model and solve using the Adams+Forrester linearization
    model1 = standard_lin(cdmkp)
    model1.optimize()
    print('solution found by model1 (standard_lin) : ' + str(model1.objVal))

    # model and solve using the Adams+Forrester linearization
    model2 = adams_and_forrester_lin(cdmkp)
    model2.optimize()
    print('solution found by model2 (adams+f_lin) : ' + str(model2.objVal))


if __name__=="__main__":
  main()

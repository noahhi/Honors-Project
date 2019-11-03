'''
Use linearizations to solve instances of the cubic Multiple knapsack to optimality
'''

from cubics import *
from gurobipy import *


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


def main():
    setParam('OutputFlag',0)
    setParam('LogFile',"")

    # generate instance of CMKP
    n = 15
    cdmkp = CMKP(n=n,density=70,m=2)

    # model and solve using the Adams+Forrester linearization
    model1 = standard_lin(cdmkp)
    model1.optimize()

    print('solution found by model1 (standard_lin) : ' + str(model1.objVal))


if __name__=="__main__":
  main()

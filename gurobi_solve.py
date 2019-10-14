from cubics import *
from gurobipy import *

'''
Apply the Adams and Forrester linearization to an instance of the cubic Multidimensional knapsack and return the model
'''
def adams_and_forrester_lin(cubic):
    # retrieve info about the instance
    n = cubic.n
    constraints = cubic.constraints
    c = cubic.c
    C = cubic.C
    D = cubic.D
    a = cubic.a
    b = cubic.b

    # create a new empty model
    m = Model()

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
    x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)
    psi = m.addVars(n, vtype=GRB.CONTINUOUS)
    tau = m.addVars(n, n, vtype=GRB.CONTINUOUS)

    for d in range(constraints):
        m.addLConstr(quicksum(x[i]*a[i][d] for i in range(n)) <= b[d])

    g = [None] * n
    for j in range(n):
        g[j] = quicksum(C[i][j]*x[i] for i in range(n))

    h = [[None] * n for i in range(n)]
    for j in range(n):
        for i in range(n):
            h[i][j] = quicksum(D[i][j][k]*x[k] for k in range(n))

    # add constraints
    for j in range(n):
        m.addConstr(psi[j] >= L_bar[j]+(x[j]*(U_bar[j]-L_bar[j]))-g[j]-sum(U_hat[i][j]*x[i]-tau[i,j] for i in range(n) if i != j))
        m.addConstr(psi[j] >= 0)
        for i in range(n):
            if i == j:
                continue
            m.addConstr(tau[i,j] >= L_hat[i][j]+(x[i]*(U_hat[i][j]-L_hat[i][j]))-h[i][j])
            m.addConstr(tau[i,j] >= 0)

    # set objective
    m.setObjective(quicksum(x[j]*(c[j]+U_bar[j])-psi[j] for j in range(n)), sense=GRB.MAXIMIZE)

    return m


'''
Apply the standard linearization to an instance of the cubic Multidimensional knapsack and return the model
'''
def standard_lin(cubic):
    n = cubic.n
    constraints = cubic.constraints
    c = cubic.c
    C = cubic.C
    D = cubic.D
    a = cubic.a
    b = cubic.b

    model = Model()

    x = model.addVars(n, name="binary_var", vtype=GRB.BINARY)
    y = model.addVars(n,n, name="quad_var", vtype=GRB.CONTINUOUS)
    z = model.addVars(n,n,n, name="cubic_var", vtype=GRB.CONTINUOUS)

    for d in range(constraints):
        model.addLConstr(quicksum(x[i]*a[i][d] for i in range(n)) <= b[d])

    for i in range(n):
        for j in range(i+1,n):
            model.addLConstr(y[i,j] <= x[i])
            model.addLConstr(y[i,j] <= x[j])
            model.addLConstr(y[i,j] >= x[i] + x[j] - 1)
            model.addLConstr(y[i,j] >= 0)
            for k in range(j+1,n):
                model.addLConstr(z[i,j,k] <= x[i])
                model.addLConstr(z[i,j,k] <= x[j])
                model.addLConstr(z[i,j,k] <= x[k])
                model.addLConstr(z[i,j,k] >= x[i] + x[j] + x[k] - 2)
                model.addLConstr(z[i,j,k] >= 0)

    linear_values = quicksum(c[i]*x[i] for i in range(n))
    quad_values = quicksum(C[i][j]*y[i,j] for i in range(n-1) for j in range(i+1,n))
    cubic_values = quicksum(D[i][j][k]*z[i,j,k] for i in range(n-2) for j in range(i+1,n-1) for k in range(j+1,n))
    model.setObjective(linear_values+quad_values+cubic_values, sense=GRB.MAXIMIZE)

    return model


def main():
    setParam('OutputFlag',0)
    setParam('LogFile',"")

    # generate instance of CMDKP
    n = 15
    cdmkp = CMDKP(n=n,density=70,constraints=1)

    # model and solve using the Adams+Forrester linearization
    model1 = standard_lin(cdmkp)
    model1.optimize()

    # model and solve using the standard linearization
    model2 = adams_and_forrester_lin(cdmkp)
    model2.optimize()

    print()

    print('solution found by model1 (standard_lin) : ' + str(model1.objVal))
    for i in range(n):
        print(model1.getVarByName("binary_var["+str(i)+"]"))

    print()

    print('solution found by model2 (adams+forr lin) : ' + str(model2.objVal))
    for i in range(n):
        print(model2.getVarByName("binary_var["+str(i)+"]"))


if __name__=="__main__":
  main()

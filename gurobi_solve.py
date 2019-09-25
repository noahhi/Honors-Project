from cubics import *
from gurobipy import *

'''
Apply the Adams and Forrester linearization to an instance of the cubic Multidimensional knapsack and return the model
'''
def adams_and_forrester_lin(cubic):
    # retrieve info about the instance
    n = cubic.n
    constraints = cubic.constraints
    P1 = cubic.P1
    P2 = cubic.P2
    P3 = cubic.P3
    w = cubic.w
    C = cubic.C

    # create a new empty model
    m = Model()

    # compute L_hat and U_hat
    L_hat = np.zeros((n,n))
    U_hat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            P3_row = P3[i][j]
            pos_indices = P3_row > 0
            U_hat[i][j] = sum(P3_row[pos_indices])
            neg_indices = P3_row < 0
            L_hat[i][j] = sum(P3_row[neg_indices])

    # compute L_bar and U_bar
    L_bar = np.zeros(n)
    U_bar = np.zeros(n)
    combined = P2 + U_hat
    for j in range(n):
        for i in range(n):
            U_bar[j] += max(combined[i][j], 0)
            L_bar[j] += min(combined[i][j], 0)

    # declare variables
    x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)
    psi = m.addVars(n, vtype=GRB.CONTINUOUS)
    tau = m.addVars(n, n, vtype=GRB.CONTINUOUS)

    # add constraints
    for j in range(n):
        g_j = LinExpr([P2[i][j] for i in range(n)],x.select())
        m.addConstr(psi[j] >= L_bar[j]+x[j]*(U_bar[j]-L_bar[j])-g_j-sum(U_hat[i][j]*x[i]-tau[i,j] for i in range(n)))
        m.addConstr(psi[j] >= 0)
        for i in range(n):
            h_ij = LinExpr([P3[i][j][k] for k in range(n)],x.select())
            m.addConstr(tau[i,j] >= L_hat[i][j]+x[i]*(U_hat[i][j]-L_hat[i][j])-h_ij)
            m.addConstr(tau[i,j] >= 0)

    # set objective
    m.setObjective(quicksum(x[j]*(P1[j]+U_bar[j])-psi[j] for j in range(n)), sense=GRB.MAXIMIZE)

    return m


'''
Apply the standard linearization to an instance of the cubic Multidimensional knapsack and return the model
'''
def standard_lin(cubic):
    n = cubic.n
    constraints = cubic.constraints
    P1 = cubic.P1
    P2 = cubic.P2
    P3 = cubic.P3
    w = cubic.w
    C = cubic.C

    model = Model()

    x = model.addVars(n, name="binary_var", vtype=GRB.BINARY)
    y = model.addVars(n,n, name="quad_var", vtype=GRB.CONTINUOUS)
    z = model.addVars(n,n,n, name="cubic_var", vtype=GRB.CONTINUOUS)

    for d in range(constraints):
        model.addLConstr(lhs=LinExpr([w[i][d] for i in range(n)],x.select()), rhs=C[d], sense=GRB.LESS_EQUAL)
        #model.addLConstr(quicksum(x[i]*w[i][d] for i in range(n)) <= C[d])

    # TODO skip when i=j?
    for i in range(n):
        for j in range(n):
            model.addLConstr(y[i,j] <= x[i])
            model.addLConstr(y[i,j] <= x[j])
            model.addLConstr(y[i,j] >= x[i] + x[j] - 1)
            model.addLConstr(y[i,j] >= 0)
            for k in range(n):
                model.addLConstr(z[i,j,k] <= x[i])
                model.addLConstr(z[i,j,k] <= x[j])
                model.addLConstr(z[i,j,k] <= x[k])
                model.addLConstr(z[i,j,k] >= x[i] + x[j] + x[k] - 2)
                model.addLConstr(z[i,j,k] >= 0)

    linear_values = quicksum(P1[i]*x[i] for i in range(n))
    quad_values = quicksum(P2[i][j]*y[i,j] for i in range(n-1) for j in range(i+1,n))
    cubic_values = quicksum(P3[i][j][k]*z[i,j,k] for i in range(n-2) for j in range(i+1,n-1) for k in range(j+1,n))
    #linear_values = LinExpr([P1[i] for i in range(n)],x.select())
    #quad_values = LinExpr([P2[i][j] for i in range(n) for j in range(n)],y.select())
    #cubic_values = LinExpr([P3[i][j][k] for i in range(n) for j in range(n) for k in range(n)],z.select())
    model.setObjective(linear_values+quad_values+cubic_values, sense=GRB.MAXIMIZE)

    return model


def main():
    setParam('OutputFlag',0)
    setParam('LogFile',"")

    # generate instance od CMDKP
    n = 4
    cdmkp = CMDKP(n=n,density=70,constraints=1)

    print()
    print(cdmkp.w)
    print()
    print(cdmkp.C)
    print()
    print(cdmkp.P1)
    print()
    print(cdmkp.P2)
    print()
    print(cdmkp.P3)

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

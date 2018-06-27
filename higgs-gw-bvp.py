from decimal import *
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import matplotlib.pyplot as plt

# Use a higher precision for decimals
getcontext().prec = 60

# Define the model parameters
param = []
param.append(Decimal('0.1')) # v0
param.append(Decimal('1')) # v1
param.append(Decimal('0.1')) # epsilon
param.append(Decimal('25951')/Decimal('1000'))# y1
param.append(Decimal('-349')/Decimal('90')) # mhsq
param.append(Decimal('2')/Decimal('9')) # lambda
param.append(Decimal('0.125')) # lambda_h
param.append(Decimal('4')) # m0sq
param.append(Decimal('-20')) # vhsq
param.append(Decimal('0.1')) # delta T

# Use the zeroth order solutions to GW and Higgs as the initial guess for BVP Solver

# Calculate the background solution for the GW field analytically. 
def GW0(y):
    """
    For a given y returns the background solution for the GW field
    """
    if type(y) == Decimal:
        yDec = y
    else:
        yDec = Decimal(str(y))
    v0, v1, epsilon, y1 = param[:4]
    GWeps = (v1 - v0*np.exp((4 - epsilon)*y1))/(np.exp(epsilon*y1) - np.exp((4 - epsilon)*y1))
    GW4 = v0 - GWeps
    res = GWeps * np.exp(epsilon*yDec) + GW4 * np.exp((4 - epsilon)*yDec)
    return res.quantize(Decimal('0.000000000000001'), rounding=ROUND_DOWN)

def GW0Der(y):
    """
    For a given y returns the derivative of the background solution for the GW field
    """
    if type(y) == Decimal:
        yDec = y
    else:
        yDec = Decimal(str(y))
    v0, v1, epsilon, y1 = param[:4]
    GWeps = (v1 - v0*np.exp((4 - epsilon)*y1))/(np.exp(epsilon*y1) - np.exp((4 - epsilon)*y1))
    GW4 = v0 - GWeps
    res = epsilon * GWeps * np.exp(epsilon*yDec) + (4 - epsilon) * GW4 * np.exp((4 - epsilon)*yDec)
    return res.quantize(Decimal('0.000000000000001'), rounding=ROUND_DOWN)

# Vectorized versions needed to create the initial guess
GW0_v = np.vectorize(GW0)
GW0Der_v = np.vectorize(GW0Der)

# Solve for the Higgs field numerically as an IVP using the background GW solution
# RHS of the first order system
def f0RHS(y, F):
    """
    Returns the RHS of the Higgs differential equation at zeroth order.
    The second order equation is written as a first order system
    """
    mhsq, lambdaGW = param[4:6]
    return (F[1], float((4 + mhsq - lambdaGW*GW0(y)))*F[0])

# Solver function
def f0Solve(tol):
    """
    Solves the Higgs differential equation at zeroth order. Raises an error when the solver fails.

    Arguments:
    tol : Tolerance of the solver. rtol = tol atol = tol**2

    Returns:
    f0Sol.t : Numpy array of mesh points
    f0Sol.y : Numpy array of f(y) and f'(y) at mesh points
    """
    f0BC = (1, float(param[7]/2 -2)) # initial conditions
    f0Sol = solve_ivp(f0RHS, [0, float(param[3])], f0BC, rtol = tol, atol = tol**2) # running the solver
    if not f0Sol.success: # check whether the solver is succesful
        raise AssertionError('solve_ivp failed with message: {mes}'.format(mes = f0Sol.message))
    else: # return the result when succesful
        return f0Sol.t, f0Sol.y


# Solve the GW+Higgs system as a BVP
# RHS of the first order system
def fRHS(y, F, p):
    """
    Returns the RHS of the first order system
    """
    hsq = p[0]
    mhsq, lambdaGW = list(map(float, param[4:6]))
    epsilon = float(param[2])
    y1 = float(param[3])
    return np.vstack((F[1], 4*F[1] + epsilon*(epsilon - 4)*F[0] - lambdaGW*hsq*np.exp(4*(y-y1))*(F[2]**2), F[3], (4 + mhsq - lambdaGW*F[0])*F[2]))

# Boundary conditions
def fBC(Fa, Fb, p):
    """
    Returns the array of boundary conditions
    """
    v0, v1 = list(map(float, param[0:2]))
    lambdaH, m0sq, vsq = list(map(float, param[6:9]))
    hsq = p[0]
    return np.array([Fa[0] - v0, Fa[2] - 1, Fa[3] - (m0sq/2 - 2)*Fa[2], Fb[0] - v1, Fb[3] + 2*Fb[2] + lambdaH*(hsq*(Fb[2]**2)-(vsq/2))*Fb[2]])

# Create the initial guess for BVP solver
def BVPGuess(tol_ivp):
    """
    Creates an initial guess for the BVP solver. 

    Arguments:
    tol_ivp : rtol values which will be used to get the zeroth order Higgs solution

    Returns:
    y : points where the initial guess is given
    F : array containing the initial guess for the functions
    h0sq : initial guess for htildesq parameter
    """
    y, F = f0Solve(tol_ivp) # Get the zeroth order Higgs solution
    FGuess = np.zeros((4,y.size)) # Create the array containing the guess
    FGuess[0] = GW0_v(y) # Guess for phi(y)
    FGuess[1] = GW0Der_v(y) # Guess for phi'(y)
    FGuess[2] = F[0] # Guess for f(y)
    FGuess[3] = F[1] # Guess for f'(y)

    # Calculate the htildesq parameter at zeroth order
    f0IR = F[0][-1]
    f0pIR = F[1][-1]
    h0sq = (float(param[8])/2. - (2 + f0pIR/f0IR)*(1./float(param[6])))*(1./(f0IR**2))

    return y, FGuess, h0sq

# BVP Solver function
def BVPSolve(tol_ivp, tol_bvp, bvp_max_nodes = 1000):
    guess = BVPGuess(tol_ivp)
    sol = solve_bvp(fRHS, fBC, guess[0], guess[1], [guess[2]], tol = tol_bvp, max_nodes = bvp_max_nodes)
    if not sol.success: # check whether the solver is succesful
        raise AssertionError('solve_bvp failed with message: {mes}'.format(mes = sol.message))
    else: # return the result when succesful
        return sol

# Plotting the solutions
yPlotMesh = np.linspace(24, float(param[3]), 200) # Mesh for plotting

BVPSol = BVPSolve(1e-5, 1e-7) # Get the BVP solution
BVPSol2 = BVPSolve(1e-9, 1e-9, bvp_max_nodes = 10000)
BVPSolFun = BVPSol.sol(yPlotMesh) # Create the interpolating function from the solutions
BVPSolFun2 = BVPSol2.sol(yPlotMesh) # Create the interpolating function from the solutions

# plt.subplot(221)
# plt.plot(yPlotMesh, BVPSolFun[0])

# plt.subplot(222)
# plt.plot(yPlotMesh, BVPSolFun[1])

# plt.subplot(223)
# plt.plot(yPlotMesh, BVPSolFun[2])

# plt.subplot(224)
# plt.plot(yPlotMesh, BVPSolFun[3])

GWratio = []

for i in range(len(yPlotMesh)):
    GWratio.append(BVPSolFun[0][i]/float(GW0(yPlotMesh[i])))

#print(GWratio)

plt.plot(yPlotMesh, GWratio , 'b-', label = 'ivp_tol = 1e-9, bvp_tol = 1e-9')
plt.legend()
plt.show()












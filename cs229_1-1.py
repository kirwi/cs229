import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def make_hypothesis_vector(xdata, theta):
    h = map(lambda xi: 1/(1 + np.exp(-np.dot(xi, theta))), xdata)
    return np.array(h)

def make_grad_vector(xdata, ydata, theta, h, m):
    grad = []
    for col in range(m):
        grad.append(((h-ydata)*xdata[:,col]).sum())
    return np.array(grad)

def hessian(xdata, ydata, theta, h, m):
    hessian = []
    for col_j in range(m):
        entry = []
        for col_k in range(m):
            entry.append( (h*(h-1)*xdata[:,col_j]*xdata[:,col_k]).sum() )
        hessian.append(entry)
    return np.array(hessian)

# Recursive Newton regression
def newton(theta):
    hypothesis = make_hypothesis_vector(xdata, theta)
    hess = hessian(xdata, ydata, theta, hypothesis, m)
    grad_l = make_grad_vector(xdata, ydata, theta, hypothesis, m)
    delta = np.linalg.solve(hess, -grad_l)
    norm_delta = np.linalg.norm(delta)
    if norm_delta < TOLERANCE:
        return theta
    return newton(theta - delta)

# Import the data. Note: Change strings to your path to the data files
xdata = np.insert(np.loadtxt('data/q1x.dat'), 0, 1., axis=1)
ydata = np.loadtxt('data/q1y.dat')

# Create Pandas data frame
df = pd.DataFrame(xdata, columns=['x0', 'x1', 'x2'])
df['y'] = ydata

# Stop recursion when within TOLERANCE of root
TOLERANCE = .000001
n, m = xdata.shape

# Parameters fit by Newton Method
initial_guess = np.zeros(m)
theta_0, theta_1, theta_2 = newton(initial_guess)

# Select out data for y=0, y=1 for plotting
y_0 = np.array(df[df.y == 0])
y_1 = np.array(df[df.y == 1])

mpl.rcParams['text.usetex'] = True

plt.figure()
plt.grid()
ax = plt.gca()
ax.set_axis_bgcolor('WhiteSmoke')
plt.title(r'q1 Data ($\theta_0$,$\theta_1$,$\theta_2$) = (%.2f, %.2f, %.2f)' %
          (theta_0, theta_1, theta_2))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

# Plot (x1,x2) data with colors indicating y=0 (blue) or y=1 (red)
plt.plot(y_0[:,1], y_0[:,2], 'bo', markersize=12, alpha=.5, label='y=0')
plt.plot(y_1[:,1], y_1[:,2], 'ro', markersize=12, alpha=.5, label='y=1')

# Plot boundary line h(x) = 0.5
x_divide = np.linspace(0,8)
y_divide = -theta_0/theta_2 - theta_1/theta_2 * x_divide
plt.plot(x_divide, y_divide, 'k-', linewidth=3, label='boundary')

plt.legend()
plt.show()

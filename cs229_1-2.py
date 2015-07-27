import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_weight_matrix(x, dm, tau):
    return np.diag(np.exp(-(x-dm.x1)**2/(2*tau**2)))

def non_weighted_theta(dm, y):
    a = np.linalg.inv(np.dot(design_matrix.T, design_matrix))
    b = np.dot(design_matrix.T, df.y)
    return np.dot(a,b)

def weighted_theta(x, dm, y, tau):
    w = make_weight_matrix(x, dm, tau)
    a = np.dot(w, dm)
    inv = np.linalg.inv(np.dot(dm.T, a))
    b = np.dot(w, y)
    c = np.dot(dm.T, b)
    return np.dot(inv, c)

# Load the data into numpy arrays
xdata = np.loadtxt('data/q2x.dat')
ydata = np.loadtxt('data/q2y.dat')

# Create the Data Frame out of a python dictionary
data = {'x0' : [1. for i in xdata], 'x1' : xdata, 'y' : ydata}
df = pd.DataFrame(data)
design_matrix = df[['x0', 'x1']]

# Some xvalues for making the fitted curves from regression
x_for_fitting = np.linspace(np.min(df.x1), np.max(df.x1), 50)

# Calculate parameters from non-weighted linear regression
linear_theta_0, linear_theta_1 = non_weighted_theta(design_matrix, df.y)

# Calculate parameters from weighted regression. This creates a theta-vector
# for each input x point. Input xpoints are the pairs [1,x] where the x come
# from x_for_fitting.
taus = [(0.1,'yellow'), (0.3,'orange'), (2,'red'), (10,'magenta')]

# Make a list of tuples (theta_vector, 'tau', 'color') for plotting
thetas= [(np.array(map(lambda x: weighted_theta(x,design_matrix,df.y,tau),
                      x_for_fitting)), str(tau), color) for tau,color in taus]

plt.figure()
plt.grid()
plt.title('Weighted/Unweighted Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(df.x1, df.y, marker='o', color='blue', markeredgecolor='k',
         markeredgewidth=1.5, markersize=13, alpha=.7, linestyle='none')
plt.plot(x_for_fitting, linear_theta_0 + linear_theta_1 * x_for_fitting, 'g-',
         linewidth=4, alpha=.8, label='unweighted')

# Plot the fitted curves for the different tau values.
for theta,label,color in thetas:
    plt.plot(x_for_fitting, theta[:,0] + theta[:,1] * x_for_fitting,
             linewidth=4, label=r'$\tau = $'+label, alpha=.8, color=color)

plt.legend(loc=4)
plt.show()

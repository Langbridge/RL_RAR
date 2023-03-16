import numpy as np
import matplotlib.pyplot as plt
from aec_env import AsyncMapEnv
import networkx as nx

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

def hill_create(map_size, height=1, width=1, centre=np.array([0., 0.]), skewness=np.array([[ 1., 0], [0,  1.]])):
    X = np.array([i for i in range(map_size)])
    Y = np.array([i for i in range(map_size)])
    X, Y = np.meshgrid(X, Y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    hill = multivariate_gaussian(pos, centre, width*skewness)
    return height*hill

def make_landscape(map_size, hill_params):
    Z = np.zeros((map_size, map_size))
    for centre, height, width in hill_params:
        Z += hill_create(map_size, height, width, np.array(centre))
    return Z

route_dict = {
              'cyclist_0': [(0, None), (8, 1), (9, 1), (17, 1), (18, 1), (26, 1), (34, 1), (42, 1), (43, 1), (51, 1), (52, 1), (53, 1), (54, 1), (62, 1), (63, 1)],
              'cyclist_1': [(0, None), (8, 1), (9, 1), (17, 1), (18, 1), (26, 1), (34, 1), (42, 1), (43, 1), (51, 1), (52, 1), (53, 1), (54, 1), (62, 1), (63, 1)]
             }

map_size = 8
map_ax = [i for i in range(map_size)]
X, Y = np.meshgrid(map_ax, map_ax)
np.random.seed(0)

# hills = [x, y], height, width
hill_attrs =  [
                [[5,2], 4, 2],
                [[3,6], 7, 3],
              ]
poll_attrs = [
                [[0,2], 7, 2],
                [[0,7], 5, 2],
                [[7,6], 6, 2],
              ]
heightmap = np.random.random_sample(size=(map_size, map_size))*0.1 + make_landscape(map_size, hill_attrs)
heightmap *= 10
pollmap = np.random.random_sample(size=(map_size, map_size))*0.1 + make_landscape(map_size, poll_attrs)
pollmap *= 25

# env = AsyncMapEnv(map_size=map_size, hill_attrs=hill_attrs, poll_attrs=poll_attrs)

for i in range(2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if i == 0:
        ax.plot_surface(X, Y, pollmap, cmap='viridis', edgecolor='none')
    else:
        ax.plot_surface(X, Y, heightmap, cmap='Greys_r', edgecolor='none')

    mapping = np.array([x for x in range(map_size**2)]).reshape(map_size, map_size)
    colours = {
                'cyclist_0': 'tab:red',
                'cyclist_1': 'tab:blue',
            }
    for agent, route in route_dict.items():
        route_x = []
        route_y = []
        route_z = []
        for d, v in route:
            route_y.append(d // map_size)
            route_x.append(d % map_size)
            route_z.append(heightmap[d // map_size, d % map_size])
        ax.plot3D(route_x, route_y, route_z, colours[agent], zorder=10)
    plt.xlabel('x')
    plt.ylabel('y')
plt.show()
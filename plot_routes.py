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

# ---- SHORTEST PATH
# route_dict = [0, 1, 9, 17, 25, 33, 41, 49, 50, 51, 52, 53, 54, 62, 63]
# pollutions = {'cyclist_0': 0.6896100808767057, 'cyclist_1': 0.2563466629830288}

# --- 7x7 OPTIMA, 50 HILL & POLLUTION RAMP
# route_dict = {'cyclist_0': [[(0, None), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1), (12, -1), (19, -1), (26, -1), (33, -1), (40, -1), (47, -1), (48, -1)], 1], 'cyclist_1': [[(0, None), (1, -1), (2, -1), (3, -1), (4, -1), (11, -1), (18, -1), (25, -1), (32, -1), (39, -1), (46, -1), (47, -1), (48, -1)], 1]}
# pollutions = {'cyclist_0': 0.1826340571736291, 'cyclist_1': 0.0788964955843816}

# --- 8x8 OPTIMA, 50 HILL & POLLUTION RAMP
# route_dict = {'less fit cyclist': [[(0, None), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1), (13, -1), (21, -1), (29, -1), (37, -1), (45, -1), (53, -1), (61, -1), (62, -1), (63, -1)], 1], 'fitter cyclist': [[(0, None), (1, -1), (2, -1), (3, -1), (11, -1), (12, -1), (20, -1), (28, -1), (36, -1), (44, -1), (52, -1), (60, -1), (61, -1), (62, -1), (63, -1)], 1]}
# pollutions = {'cyclist_0': 0.20252626233171678, 'cyclist_1': 0.08210437610887333}

# --- 10 AGENT TRAIN, GAMMA, NO VEL, HILLS
route_dict = {'less fit cyclist': [[(0,), (8,), (16,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,)], 1], 'fitter cyclist': [[(8,), (16,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,), (24,), (32,)],1]}
pollutions = {'cyclist_0': 1.6392214531183604, 'cyclist_1': 3.240597021933907}

map_size = 8
plt_map_size = 8

map_ax = [i for i in range(map_size)]
X, Y = np.meshgrid(map_ax, map_ax)
np.random.seed(0)

# hills = [x, y], height, width
hill_attrs =  [
                # [[5,2], 4, 2],
                # [[3,6], 7, 3],
                [[3, 3], 50, 2],
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
        plt_map = pollmap
        print('\nMean pollution:', end="\t")
    else:
        plt_map = heightmap
        print('\nMean height:', end="\t")
        
    ax.plot_surface(X, Y, plt_map, cmap='Greys_r', edgecolor='none')

    mapping = np.array([x for x in range(map_size**2)]).reshape(map_size, map_size)
    colours = {
                'less fit cyclist': 'tab:red',
                'fitter cyclist': 'tab:blue',
            }
    for agent, route in route_dict.items():
        route_x = []
        route_y = []
        route_z = []
        for attrs in route[0]:
            d = attrs[0]
            route_y.append(d // plt_map_size)
            route_x.append(d % plt_map_size)
            route_z.append(plt_map[d // plt_map_size, d % plt_map_size])
        print(f'{agent} {np.mean(route_z)}', end="\t")
        ax.plot3D(route_x, route_y, route_z, colours[agent], zorder=10, label=f'{agent}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
plt.show()
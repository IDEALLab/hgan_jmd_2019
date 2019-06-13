"""
Plots samples or new shapes in the semantic space.

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
"""

from matplotlib import pyplot as plt
#plt.switch_backend('agg')
import numpy as np
from utils import gen_grid


def plot_shape(xys, z1, z2, ax, scale, scatter, mirror, **kwargs):
#    mx = max([y for (x, y) in m])
#    mn = min([y for (x, y) in m])
    xscl = scale# / (mx - mn)
    yscl = scale# / (mx - mn)
#    ax.scatter(z1, z2)
    if 'c' not in kwargs:
        kwargs['c'] = 'b'
    if scatter:
        ax.scatter( *zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), edgecolors='none', s=1.5, **kwargs)
    else:
        ax.plot( *zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), **kwargs)
        if mirror == 'y':
            ax.plot( *zip(*[(-x * xscl + z1, y * yscl + z2) for (x, y) in xys]), lw=.2, c='b')
    #        plt.fill_betweenx( *zip(*[(y * yscl + z2, -x * xscl + z1, x * xscl + z1)
    #                          for (x, y) in xys]), color=kwargs['color'], alpha=kwargs['alpha']*.8)
        elif mirror == 'x':
            ax.plot( *zip(*[(x * xscl + z1, -y * yscl + z2) for (x, y) in xys]), lw=.2, c='b')
    #        plt.fill_betweenx( *zip(*[(y * yscl + z2, -x * xscl + z1, x * xscl + z1)
    #                          for (x, y) in xys]), color=kwargs['color'], alpha=kwargs['alpha']*.8)

def plot_samples(Z, X_list, alphas=None, scale=None, scatter=True, 
                 mirror=False, annotate=False, fname=None, **kwargs):
    
    ''' Plot shapes given design sapce and latent space coordinates '''
    
    plt.rc("font", size=12)
    
    if Z is None or Z.shape[1] != 2:
        N = X_list[0].shape[0]
        points_per_axis = int(N**.5)
        bounds = (-1., 1.)
        Z = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid
        scale = 0.8*2.0/points_per_axis

    if scale is None:
        r0 = np.max(Z[:,0]) - np.min(Z[:,0])
        r1 = np.max(Z[:,1]) - np.min(Z[:,1])
        scale = 0.8*np.minimum(r0, r1)/10
        
    # Create a 2D plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    kwargs2 = kwargs.copy()
    if 'samples' not in fname and 'assemblies' not in fname:
        kwargs2['c'] = 'gray'
    for (i, z) in enumerate(Z):
        if alphas is not None:
            kwargs['alpha'] = alphas[i]
            kwargs2['alpha'] = alphas[i]
        for (j, X) in enumerate(X_list):
            if j == 0:
                plot_shape(X[i], z[0], z[1], ax, scale, scatter, mirror, **kwargs)
            else:
                plot_shape(X[i], z[0], z[1], ax, scale, scatter, mirror, **kwargs2)
            if annotate:
                label = '{0}'.format(i+1)
                plt.annotate(label, xy = (z[0], z[1]), size=10)
    
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(fname+'.svg', dpi=300)
#    plt.savefig(fname+'.png', dpi=300)
    plt.close()

def plot_synthesized(Z, gen_func, proba_func=None, d=2, scale=.8, scatter=True, mirror=False, fname=None, **kwargs):
    
    ''' Synthesize shapes given latent space coordinates and plot them '''
    
    if d <= 3:
        latent = Z[:,:d]
    else:
        latent = np.random.normal(scale=0.5, size=(Z.shape[0], d))
    X_list = gen_func(latent)
    
    alphas = None
    if proba_func is not None:
        proba = proba_func(*X_list)
        alphas = 0.3 + 0.7*proba # rescale alpha to [0.3, 1.0]
    
    plot_samples(Z, X_list, alphas, scale, scatter, mirror, fname=fname, **kwargs)

def plot_grid(points_per_axis, gen_func, proba_func=None, d=2, bounds=(0.0, 1.0), scale=.8, 
              scatter=False, mirror=False, fname=None, **kwargs):
    
    ''' Uniformly plots synthesized shapes in the latent space
        K : number of samples for each point in the latent space '''
        
    scale *= (bounds[1]-bounds[0])/points_per_axis
    
    if d == 1:
        Z = np.linspace(bounds[0], bounds[1], points_per_axis)
        Z = np.vstack((Z, np.zeros(points_per_axis))).T
        plot_synthesized(Z, gen_func, proba_func, 1, scale, scatter, mirror, fname, **kwargs)
    if d == 2:
        Z = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid
        plot_synthesized(Z, gen_func, proba_func, 2, scale, scatter, mirror, fname, **kwargs)
    if d >= 3:
        Z = np.random.uniform(bounds[0], bounds[1], (points_per_axis**3, d))
        Z[:, :3] = gen_grid(3, points_per_axis, bounds[0], bounds[1]) # Generate a grid
        X_list = gen_func(Z)
        alphas = None
        if proba_func is not None:
            proba = proba_func(*X_list)
            alphas = 0.3 + 0.7*proba # rescale alpha to [0.3, 1.0]
        zgrid = np.linspace(bounds[0], bounds[1], points_per_axis)
        for i in range(points_per_axis):
            Zxy = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid
            ind = Z[:,2]==zgrid[i]
            Xi_list = []
            for X in X_list:
                Xi = X[ind]
                Xi_list.append(Xi)
            plot_samples(Zxy, Xi_list, alphas, scale, scatter, mirror, fname='%s_%.2f' % (fname, zgrid[i]), **kwargs)


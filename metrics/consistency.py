#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 20:06:02 2019

@author: weichen
"""

from importlib import import_module
import numpy as np
import matplotlib.pyplot as plt
from utils import mean_err

def sample_cartesian_basis(d, m, bounds):
    # Sample m points along a line parallel to a d-dimensional space's basis
    basis = np.random.choice(d)
    c = np.zeros((m, d))
    c[:,:] = np.random.uniform(bounds[0], bounds[1], d)
    c[:,basis] = np.random.uniform(bounds[0], bounds[1], m)
    return c

def sample_polar_basis(m, bounds):
    # Sample m points along a d-dimensional space's polar basis
    basis = np.random.choice(2)
    c = np.zeros((m, 2))
    if basis == 1: # selected basis is the radius
        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(bounds[0], bounds[1], m)
    else: # selected basis is the polar angle
        theta = np.sort(np.random.uniform(0, 2*np.pi, m))
        r = np.random.uniform(bounds[0], bounds[1])
    c[:,0] = r * np.cos(theta)
    c[:,1] = r * np.sin(theta)
    return c

def consistency(gen_func, d, bounds, parents=None, basis='cartesian'):
    
    n_eval = 100
    n_points = 50
    mean_cor = 0
    
    for i in range(n_eval):
        
        ind = np.arange(n_points)
        np.random.shuffle(ind)
        
        if basis == 'polar':
            c = sample_polar_basis(n_points, bounds)
        else:
            c = sample_cartesian_basis(d, n_points, bounds)
        dist_c = np.linalg.norm(c-c[ind], axis=1)
        
        if parents is not None:
            out = gen_func(c, parents)
        else:
            out = gen_func(c)
        if type(out) == list:
            X = out[0]
        else:
            X = out
        X = X.reshape((n_points, -1))
        dist_X = np.linalg.norm(X-X[ind], axis=1)
        
        mean_cor += np.corrcoef(dist_c, dist_X)[0,1]
        
    return mean_cor/n_eval

def ci_cons(n, gen_func, d=2, bounds=(0.0, 1.0), basis='cartesian'):
    conss = np.zeros(n)
    for i in range(n):
        conss[i] = consistency(gen_func, d, bounds, basis=basis)
    mean, err = mean_err(conss)
    return mean, err


if __name__ == '__main__':
    
    latent_dim = [2, 2, 2]
    noise_dim = [10, 0, 0]
    n_points = [64, 64, 64]
    bezier_degree = [31, None, None]
    bounds = (0.0, 1.0)
    
    sample_sizes = [500, 2000, 4000, 7000, 10000]
    for sample_size in sample_sizes:
        h = import_module('SCC.hgan')
        model = h.Model(latent_dim, noise_dim, n_points, bezier_degree)
        results_dir = 'results/SCC/hgan/{}'.format(sample_size)
        model.restore(save_dir=results_dir)
        
        n_runs = 10
        cons1_mean, cons1_err = ci_cons(n_runs, model.synthesize_x1, latent_dim[1], bounds)
        cons2_mean, cons2_err = ci_cons(n_runs, model.synthesize_x2, latent_dim[2], bounds)
        
        results_mesg_4 = 'Consistency for latent space 1: %.3f +/- %.3f' % (cons1_mean, cons1_err)
        results_mesg_5 = 'Consistency for latent space 2: %.3f +/- %.3f' % (cons2_mean, cons2_err)
        
        print(results_mesg_4)
        print(results_mesg_5)

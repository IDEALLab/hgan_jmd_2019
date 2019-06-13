"""
Trains a GAN, and visulizes results

Author(s): Wei Chen (wchen459@umd.edu)
"""

import argparse
import os.path
import json
import numpy as np
from importlib import import_module

from shape_plot import plot_samples, plot_grid
from metrics.diversity import ci_rdiv
from metrics.mmd import ci_mmd
from metrics.precision import ci_prc
from metrics.consistency import ci_cons
from utils import ElapsedTimer


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', type=str, default='startover', help='startover, continue, or evaluate')
    parser.add_argument('data', type=str, default='AHH', help='AHH, SEoEi, SEiEo, or SCC')
    parser.add_argument('--sample_size', type=int, default=10000, help='sample size')
    parser.add_argument('--save_interval', type=int, default=500, help='save interval')
    args = parser.parse_args()
    assert args.mode in ['startover', 'continue', 'evaluate']
    assert args.data in ['AHH', 'SEoEi', 'SEiEo', 'SCC', 'AH', 'SE', 'SC', 'SEoEmEi', 'SCCC', 'S', 'SEoEmEmEi', 'SCCCC']
    
    print('##################################################################')
    print('Data: {}'.format(args.data))
    model_name = 'infogan'
    print('Model: {}'.format(model_name))
    print('Sample size: {}'.format(args.sample_size))
    
    # Set hyper-parameters
    if args.data == 'AHH':
        data_fname = 'results/AHH/AHH.npy'
        latent_dim = 8
        noise_dim = 10
        n_points = [64, 64, 64]
        dependency = [[], [0], [0,1]]
        bezier_degree = [31, None, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif args.data == 'SEoEi':
        data_fname = 'results/SEoEi/SEoEi.npy'
        latent_dim = 6
        noise_dim = 10
        n_points = [64, 64, 64]
        dependency = [[], [0], [1]]
        bezier_degree = [31, None, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif args.data == 'SEiEo':
        data_fname = 'results/SEiEo/SEiEo.npy'
        latent_dim = 6
        noise_dim = 10
        n_points = [64, 64, 64]
        dependency = [[], [0], [0,1]]
        bezier_degree = [31, None, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif args.data == 'SCC':
        data_fname = 'results/SCC/SCC.npy'
        latent_dim = 6
        noise_dim = 10
        n_points = [64, 64, 64]
        dependency = [[], [0], [0]]
        bezier_degree = [31, None, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif args.data == 'AH':
        data_fname = 'results/AH/AH.npy'
        latent_dim = 6
        noise_dim = 10
        n_points = [64, 64]
        bezier_degree = [31, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif args.data == 'SE':
        data_fname = 'results/SE/SE.npy'
        latent_dim = 4
        noise_dim = 10
        n_points = [64, 64]
        bezier_degree = [31, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif args.data == 'SC':
        data_fname = 'results/SC/SC.npy'
        latent_dim = 4
        noise_dim = 10
        n_points = [64, 64]
        bezier_degree = [31, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif  args.data == 'SEoEmEi':
        data_fname = 'results/SEoEmEi/SEoEmEi.npy'
        latent_dim = 8
        noise_dim = 10
        n_points = [64, 64, 64, 64]
        bezier_degree = [31, None, None, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif  args.data == 'SCCC':
        data_fname = 'results/SCCC/SCCC.npy'
        latent_dim = 8
        noise_dim = 10
        n_points = [64, 64, 64, 64]
        bezier_degree = [31, None, None, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif args.data == 'S':
        data_fname = 'results/S/S.npy'
        latent_dim = 2
        noise_dim = 10
        n_points = [64]
        bezier_degree = [31]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif  args.data == 'SEoEmEi':
        data_fname = 'results/SEoEmEi/SEoEmEi.npy'
        latent_dim = 10
        noise_dim = 10
        n_points = [64, 64, 64, 64, 64]
        bezier_degree = [31, None, None, None, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif  args.data == 'SCCC':
        data_fname = 'results/SCCC/SCCC.npy'
        latent_dim = 10
        noise_dim = 10
        n_points = [64, 64, 64, 64, 64]
        bezier_degree = [31, None, None, None, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif  args.data == 'SEoEmEmEi':
        data_fname = 'results/SEoEmEmEi/SEoEmEmEi.npy'
        latent_dim = 10
        noise_dim = 10
        n_points = [64, 64, 64, 64, 64]
        bezier_degree = [31, None, None, None, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif  args.data == 'SCCCC':
        data_fname = 'results/SCCCC/SCCCC.npy'
        latent_dim = 10
        noise_dim = 10
        n_points = [64, 64, 64, 64, 64]
        bezier_degree = [31, None, None, None, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    
    # Read dataset
    X = np.load(data_fname)
    ind = np.random.choice(X.shape[0], size=args.sample_size, replace=False)
    X = X[ind]
    
    # Split training and test data
    test_split = 0.8
    N = X.shape[0]
    split = int(N*test_split)
    X_train = X[:split]
    X_test = X[split:]
    
    results_dir = 'results/{}/{}/{}'.format(args.data, model_name, args.sample_size)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Train
    h = import_module('{}.{}'.format(args.data, model_name))
    model = h.Model(latent_dim, noise_dim, n_points, bezier_degree)
    if args.mode == 'startover':
        timer = ElapsedTimer()
        model.train(X_train, batch_size=batch_size, train_steps=train_steps, save_interval=args.save_interval, save_dir=results_dir)
        elapsed_time = timer.elapsed_time()
        runtime_mesg = 'Wall clock time for training: %s' % elapsed_time
        print(runtime_mesg)
        runtime_file = open('{}/runtime.txt'.format(results_dir), 'w')
        runtime_file.write('%s\n' % runtime_mesg)
        runtime_file.close()
    else:
        model.restore(save_dir=results_dir)
    
    print('Plotting training samples ...')
    samples_list = []
    n_points_c = np.cumsum([0,] + n_points)
    for i in range(len(n_points)):
        samples_list.append(np.squeeze(X)[:25, n_points_c[i]:n_points_c[i+1]])
    plot_samples(None, samples_list, scatter=False, alpha=.7, c='k', fname='%s/samples' % args.data)
        
    print('Plotting synthesized assemblies ...')
    assemblies_list = model.synthesize(25)
    plot_samples(None, assemblies_list, scatter=False, alpha=.7, c='k', fname='{}/assemblies'.format(results_dir))
    
    print('Plotting synthesized assemblies in order ...')
    def synthesize(X_latent, dims=[0,1]):
        X_latent_full = np.zeros((X_latent.shape[0], latent_dim))
        X_latent_full[:, dims] = X_latent
        return model.synthesize(X_latent_full)
    
    for i in range(0, latent_dim, 2):
        gen_func = lambda x: synthesize(x, [i,i+1])
        plot_grid(5, gen_func=gen_func, d=2, scale=.95, 
                  scatter=False, alpha=.7, c='k', fname='{}/assemblies{}'.format(results_dir, i/2))
    
    n_runs = 10
    
    feasibility_func = import_module('{}.feasibility'.format(args.data)).check_feasibility
    prc_mean, prc_err = ci_prc(n_runs, model.synthesize_assemblies, feasibility_func, n_points)
    print('Precision for assembly: %.3f +/- %.3f' % (prc_mean, prc_err))
    mmd_mean, mmd_err = ci_mmd(n_runs, model.synthesize_assemblies, X_test)
    rdiv_mean, rdiv_err = ci_rdiv(n_runs, X_test, model.synthesize_assemblies)
    basis = 'cartesian'
    cons_mean, cons_err = ci_cons(n_runs, model.synthesize_assemblies, latent_dim, bounds, basis=basis)
    
    res = {'CSS': [prc_mean, prc_err], 
           'MMD': [mmd_mean, mmd_err], 
           'R-Div': [rdiv_mean, rdiv_err], 
           'LSC': [cons_mean, cons_err]}
    json.dump(res, open('{}/results.json'.format(results_dir), 'w'))
    
    results_mesg_0 = 'Precision for assembly: %.3f +/- %.3f' % (prc_mean, prc_err)
    results_mesg_1 = 'Maximum mean discrepancy for assembly: %.4f +/- %.4f' % (mmd_mean, mmd_err)
    results_mesg_2 = 'Relative diversity for assembly: %.3f +/- %.3f' % (rdiv_mean, rdiv_err)
    results_mesg_3 = 'Consistency for latent space: %.3f +/- %.3f' % (cons_mean, cons_err)
        
    results_file = open('{}/results.txt'.format(results_dir), 'w')
    
    print(results_mesg_0)
    results_file.write('%s\n' % results_mesg_0)
    print(results_mesg_1)
    results_file.write('%s\n' % results_mesg_1)
    print(results_mesg_2)
    results_file.write('%s\n' % results_mesg_2)
    print(results_mesg_3)
    results_file.write('%s\n' % results_mesg_3)
        
    results_file.close()


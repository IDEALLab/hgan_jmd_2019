"""
Trains an HGAN, and visulizes results

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
    parser.add_argument('--model', type=str, default='hgan', help='hgan, hgan_cat, or hgan_wo_info')
    parser.add_argument('--sample_size', type=int, default=10000, help='sample size')
    parser.add_argument('--save_interval', type=int, default=500, help='save interval')
    args = parser.parse_args()
    assert args.mode in ['startover', 'continue', 'evaluate']
    assert args.data in ['SEoEmEi', 'SCCC']
    assert args.model in ['hgan']
    
    print('##################################################################')
    print('Data: {}'.format(args.data))
    print('Model: {}'.format(args.model))
    print('Sample size: {}'.format(args.sample_size))
    
    # Set hyper-parameters
    if  args.data == 'SEoEmEi':
        data_fname = 'results/SEoEmEi/SEoEmEi.npy'
        latent_dim = [2, 2, 2, 2]
        noise_dim = [10, 0, 0, 0]
        n_points = [64, 64, 64, 64]
        dependency = [[], [0], [1], [2]]
        bezier_degree = [31, None, None, None]
        bounds = (0.0, 1.0)
        train_steps = 100000
        batch_size = 32
    elif  args.data == 'SCCC':
        data_fname = 'results/SCCC/SCCC.npy'
        latent_dim = [2, 2, 2, 2]
        noise_dim = [10, 0, 0, 0]
        n_points = [64, 64, 64, 64]
        dependency = [[], [0], [0], [0]]
        bezier_degree = [31, None, None, None]
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
    
    X0_test = X_test[:,:n_points[0]]
    X1_test = X_test[:,n_points[0]:n_points[0]+n_points[1]]
    X2_test = X_test[:,n_points[0]+n_points[1]:-n_points[3]]
    X3_test = X_test[:,-n_points[3]:]
    X_test_list = [X0_test, X1_test, X2_test, X3_test]
    
    results_dir = 'results/{}/{}/{}'.format(args.data, args.model, args.sample_size)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Train
    h = import_module('{}.{}'.format(args.data, args.model))
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
    
    print('Plotting synthesized x0 ...')
    plot_grid(5, gen_func=model.synthesize_x0, d=latent_dim[0], scale=.95, 
              scatter=False, alpha=.7, c='k', fname='{}/x0'.format(results_dir))
    
    X2, X0, X1 = model.synthesize_x2(1)
    
    print('Plotting synthesized x1 ...')
    synthesize_x1 = lambda x: model.synthesize_x1(x, [X0])
    plot_grid(5, gen_func=synthesize_x1, d=latent_dim[1], scale=.95, 
              scatter=False, alpha=.7, c='k', fname='{}/x1'.format(results_dir))
    
    print('Plotting synthesized x2 ...')
    synthesize_x2 = lambda x: model.synthesize_x2(x, [X0, X1])
    plot_grid(5, gen_func=synthesize_x2, d=latent_dim[2], scale=.95, 
              scatter=False, alpha=.7, c='k', fname='{}/x2'.format(results_dir))
    
    print('Plotting synthesized x3 ...')
    synthesize_x3 = lambda x: model.synthesize_x3(x, [X0, X1, X2])
    plot_grid(5, gen_func=synthesize_x3, d=latent_dim[3], scale=.95, 
              scatter=False, alpha=.7, c='k', fname='{}/x3'.format(results_dir))
        
    print('Plotting synthesized assemblies ...')
    assemblies = model.synthesize_assemblies(25)
    assemblies_list = []
    for i in range(len(n_points)):
        assemblies_list.append(assemblies[:25, n_points_c[i]:n_points_c[i+1]])
    plot_samples(None, assemblies_list, scatter=False, alpha=.7, c='k', fname='{}/assemblies'.format(results_dir))
    
    n_runs = 10
    
    feasibility_func = import_module('{}.feasibility'.format(args.data)).check_feasibility
    prc_mean, prc_err = ci_prc(n_runs, model.synthesize_assemblies, feasibility_func, n_points)
    print('Precision for assembly: %.3f +/- %.3f' % (prc_mean, prc_err))
    mmd_mean, mmd_err = ci_mmd(n_runs, model.synthesize_assemblies, X_test)
    rdiv_mean, rdiv_err = ci_rdiv(n_runs, X_test, model.synthesize_assemblies)
    if args.data == 'SCCC':
        basis = 'polar'
    else:
        basis = 'cartesian'
    cons0_mean, cons0_err = ci_cons(n_runs, model.synthesize_x0, latent_dim[0], bounds, basis='cartesian')
    cons1_mean, cons1_err = ci_cons(n_runs, model.synthesize_x1, latent_dim[1], bounds, basis=basis)
    cons2_mean, cons2_err = ci_cons(n_runs, model.synthesize_x2, latent_dim[2], bounds, basis=basis)
    cons3_mean, cons3_err = ci_cons(n_runs, model.synthesize_x3, latent_dim[3], bounds, basis=basis)
    
    res = {'CSS': [prc_mean, prc_err], 
           'MMD': [mmd_mean, mmd_err], 
           'R-Div': [rdiv_mean, rdiv_err], 
           'LSC_A': [cons0_mean, cons0_err], 
           'LSC_B': [cons1_mean, cons1_err], 
           'LSC_C': [cons2_mean, cons2_err], 
           'LSC_D': [cons3_mean, cons3_err]}
    json.dump(res, open('{}/results.json'.format(results_dir), 'w'))
    
    results_mesg_0 = 'Precision for assembly: %.3f +/- %.3f' % (prc_mean, prc_err)
    results_mesg_1 = 'Maximum mean discrepancy for assembly: %.4f +/- %.4f' % (mmd_mean, mmd_err)
    results_mesg_2 = 'Relative diversity for assembly: %.3f +/- %.3f' % (rdiv_mean, rdiv_err)
    results_mesg_3 = 'Consistency for latent space 0: %.3f +/- %.3f' % (cons0_mean, cons0_err)
    results_mesg_4 = 'Consistency for latent space 1: %.3f +/- %.3f' % (cons1_mean, cons1_err)
    results_mesg_5 = 'Consistency for latent space 2: %.3f +/- %.3f' % (cons2_mean, cons2_err)
    results_mesg_6 = 'Consistency for latent space 3: %.3f +/- %.3f' % (cons3_mean, cons3_err)
        
    results_file = open('{}/results.txt'.format(results_dir), 'w')
    
    print(results_mesg_0)
    results_file.write('%s\n' % results_mesg_0)
    print(results_mesg_1)
    results_file.write('%s\n' % results_mesg_1)
    print(results_mesg_2)
    results_file.write('%s\n' % results_mesg_2)
    print(results_mesg_3)
    results_file.write('%s\n' % results_mesg_3)
    print(results_mesg_4)
    results_file.write('%s\n' % results_mesg_4)
    print(results_mesg_5)
    results_file.write('%s\n' % results_mesg_5)
    print(results_mesg_6)
    results_file.write('%s\n' % results_mesg_6)
        
    results_file.close()


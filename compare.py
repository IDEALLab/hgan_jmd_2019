
"""
Compare preformance of methods within certain running time.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import argparse
import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', type=str, default='train', help='train, evaluate, or plot')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate', 'plot']
    
    sample_sizes = [500, 2000, 4000, 7000, 10000]
    
    metrics = ['CSS', 'MMD', 'R-Div', 'LSC_A', 'LSC_B', 'LSC_C']
    examples = ['AHH', 'SEoEi', 'SEiEo', 'SCC']
    examples2 = ['AH', 'SE']
    
    ###########################################################################
    model = 'hgan'
    
    ''' Train/Evaluate '''
    if args.mode == 'train':
        for example in examples+examples2:
            for sample_size in sample_sizes:
                if not os.path.exists('results/{}/{}/{}/results.json'.format(example, model, sample_size)):
                    if example in examples:
                        os.system('python run_3parts.py startover {} --model={} --sample_size={} --save_interval={}'.format(example, model, sample_size, 
                                  args.save_interval))
                    else:
                        os.system('python run_2parts.py startover {} --model={} --sample_size={} --save_interval={}'.format(example, model, sample_size, 
                                  args.save_interval))
    elif args.mode == 'evaluate':
        for example in examples+examples2:
            for sample_size in sample_sizes:
                if example in examples:
                    os.system('python run_3parts.py evaluate {} --model={} --sample_size={} --save_interval={}'.format(example, model, sample_size, 
                              args.save_interval))
                else:
                    os.system('python run_2parts.py evaluate {} --model={} --sample_size={} --save_interval={}'.format(example, model, sample_size, 
                              args.save_interval))
    
    ''' Plot '''
    plt.figure(figsize=(10, 6))
    positions = [231, 232, 233, 234, 235, 236]
    
    for j, metric in enumerate(metrics):
        plt.subplot(positions[j])
        fmts = ['-o', '--o', ':o', '-.o', '-s', '-^']
        fmts_cycle = itertools.cycle(fmts)
        if metric == 'LSC_C':
            example_list = examples
        else:
            example_list = examples+examples2
        for example in example_list:
            mean_err = np.zeros((2, len(sample_sizes)))
            for i, sample_size in enumerate(sample_sizes):
                mean_err[:,i] = json.load(open('results/{}/{}/{}/results.json'.format(example, model, sample_size)))[metric]
            plt.errorbar(sample_sizes, mean_err[0], yerr=mean_err[1], fmt=next(fmts_cycle), label=example, alpha=0.8)
        plt.legend()
        plt.xlabel('Sample Size')
        plt.ylabel(metric)
        
    plt.tight_layout()
    plt.savefig('results/metrics.svg')
    plt.close()
    
    ###########################################################################
#    model = 'hgan_wo_info'
#    example = 'SEoEi'
#    
#    if example in ['SE', 'AH']:
#        metrics = ['LSC_A', 'LSC_B']
#    else:
#        metrics = ['LSC_A', 'LSC_B', 'LSC_C']
#    
#    ''' Train/Evaluate '''
#    if args.mode == 'train':
#        for sample_size in sample_sizes:
#            if not os.path.exists('results/{}/{}/{}/results.json'.format(example, model, sample_size)):
#                if example in examples:
#                    os.system('python run_3parts.py startover {} --model={} --sample_size={} --save_interval={}'.format(example, model, sample_size, 
#                              args.save_interval))
#                else:
#                    os.system('python run_2parts.py startover {} --model={} --sample_size={} --save_interval={}'.format(example, model, sample_size, 
#                              args.save_interval))
#                
#    elif args.mode == 'evaluate':
#        for sample_size in sample_sizes:
#            if example in examples:
#                os.system('python run_3parts.py evaluate {} --model={} --sample_size={} --save_interval={}'.format(example, model, sample_size, 
#                          args.save_interval))
#            else:
#                os.system('python run_2parts.py evaluate {} --model={} --sample_size={} --save_interval={}'.format(example, model, sample_size, 
#                          args.save_interval))
#        
#    ''' Plot '''
#    plt.figure(figsize=(10, 3))
#    positions = [231, 232, 233, 234, 235, 236]
#    
#    for j, metric in enumerate(metrics):
#        plt.subplot(positions[j])
#        fmts = ['--o', '-^']
#        fmts_cycle = itertools.cycle(fmts)
#        for model in ['hgan', 'hgan_wo_info']:
#            mean_err = np.zeros((2, len(sample_sizes)))
#            for i, sample_size in enumerate(sample_sizes):
#                mean_err[:,i] = json.load(open('results/{}/{}/{}/results.json'.format(example, model, sample_size)))[metric]
#            plt.errorbar(sample_sizes, mean_err[0], yerr=mean_err[1], fmt=next(fmts_cycle), label=model, alpha=0.8)
#        plt.legend()
#        plt.xlabel('Sample Size')
#        plt.ylabel(metric)
#        
#    plt.tight_layout()
#    plt.savefig('results/{}_LSC.svg'.format(example))
#    plt.close()
    
    ###########################################################################
    examples_depth = ['S', 'SE', 'SEoEi', 'SEoEmEi', 'SEoEmEmEi']
    examples_breadth = ['S', 'SC', 'SCC', 'SCCC', 'SCCCC']
    metrics = ['CSS', 'MMD', 'R-Div']
    
    ''' Train/Evaluate '''
    if args.mode == 'train':
        for example in examples_depth+examples_breadth:
            if not os.path.exists('results/{}/hgan/10000/results.json'.format(example)) and example != 'S':
                if example in ['SE', 'SC']:
                    os.system('python run_2parts.py startover {} --save_interval={}'.format(example, args.save_interval))
                elif example in ['SEoEi', 'SCC']:
                    os.system('python run_3parts.py startover {} --save_interval={}'.format(example, args.save_interval))
                elif example in ['SEoEmEi', 'SCCC']:
                    os.system('python run_4parts.py startover {} --save_interval={}'.format(example, args.save_interval))
                elif example in ['SEoEmEmEi', 'SCCCC']:
                    os.system('python run_5parts.py startover {} --save_interval={}'.format(example, args.save_interval))
            if not os.path.exists('results/{}/infogan/10000/results.json'.format(example)):
                os.system('python run_infogan.py startover {} --save_interval={}'.format(example, args.save_interval))
    elif args.mode == 'evaluate':
        for example in examples_depth+examples_breadth:
            if example in ['SE', 'SC']:
                os.system('python run_2parts.py evaluate {} --save_interval={}'.format(example, args.save_interval))
            elif example in ['SEoEi', 'SCC']:
                os.system('python run_3parts.py evaluate {} --save_interval={}'.format(example, args.save_interval))
            elif example in ['SEoEmEi', 'SCCC']:
                os.system('python run_4parts.py evaluate {} --save_interval={}'.format(example, args.save_interval))
            elif example in ['SEoEmEmEi', 'SCCCC']:
                os.system('python run_5parts.py evaluate {} --save_interval={}'.format(example, args.save_interval))
            os.system('python run_infogan.py evaluate {} --save_interval={}'.format(example, args.save_interval))
            
    ''' Plot '''
    plt.figure(figsize=(10, 6))
    positions_depth = [231, 232, 233]
    positions_breadth = [234, 235, 236]
            
    def draw_subplot(positions, examples):
        x = range(len(examples))
        for j, metric in enumerate(metrics):
            plt.subplot(positions[j])
            fmts = ['-o', '--^']
            fmts_cycle = itertools.cycle(fmts)
            for model in ['hgan', 'infogan']:
                mean_err = np.zeros((2, len(examples)))
                for i, example in enumerate(examples):
                    if example == 'S' and model == 'hgan':
                        mean_err[:,i] = json.load(open('results/{}/infogan/10000/results.json'.format(example)))[metric]
                    else:
                        mean_err[:,i] = json.load(open('results/{}/{}/10000/results.json'.format(example, model)))[metric]
                plt.errorbar(x, mean_err[0], yerr=mean_err[1], fmt=next(fmts_cycle), label=model, alpha=0.8)
            plt.xticks(x, examples)
            plt.legend()
            plt.xlabel('Example')
            plt.ylabel(metric)
            
    draw_subplot(positions_depth, examples_depth)
    draw_subplot(positions_breadth, examples_breadth)
    plt.tight_layout()
    plt.savefig('results/depth_breadth.svg')
    plt.close()
                
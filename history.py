"""
Compare training history.

Author(s): Wei Chen (wchen459@umd.edu)
"""


import numpy as np
import matplotlib.pyplot as plt


def subplot_history(position, history_hgan, history_infogan, title):
    plt.subplot(position)
    plt.plot(history_hgan[:,0], history_hgan[:,1], 'b-', alpha=0.7, label='HGAN')
    plt.plot(history_infogan[:,0], history_infogan[:,1], 'r--', alpha=0.7, label='InfoGAN')
    plt.legend()
    plt.title(title)
    

#examples = ['AHH', 'SEoEi', 'SEiEo', 'SCC', 'AH', 'SE']
examples = ['SEoEi', 'SEoEmEmEi']

for example in examples:
    
    save_dir = './results/{}'.format(example)
    plt.figure(figsize=(12, 3))
    
    d_fake_hgan = np.genfromtxt('{}/hgan/10000/logs/run_.-tag-D_loss_for_fake.csv'.format(save_dir), 
                                delimiter=',')[1:, 1:]
    d_fake_infogan = np.genfromtxt('{}/infogan/10000/logs/run_.-tag-D_loss_for_fake.csv'.format(save_dir), 
                                   delimiter=',')[1:, 1:]
    subplot_history(131, d_fake_hgan, d_fake_infogan, 'D loss for generated data')
    
    d_real_hgan = np.genfromtxt('{}/hgan/10000/logs/run_.-tag-D_loss_for_real.csv'.format(save_dir), 
                                delimiter=',')[1:, 1:]
    d_real_infogan = np.genfromtxt('{}/infogan/10000/logs/run_.-tag-D_loss_for_real.csv'.format(save_dir), 
                                   delimiter=',')[1:, 1:]
    subplot_history(132, d_real_hgan, d_real_infogan, 'D loss for real data')
    
    g_hgan = np.genfromtxt('{}/hgan/10000/logs/run_.-tag-G_loss.csv'.format(save_dir), 
                           delimiter=',')[1:, 1:]
    g_infogan = np.genfromtxt('{}/infogan/10000/logs/run_.-tag-G_loss.csv'.format(save_dir), 
                              delimiter=',')[1:, 1:]
    subplot_history(133, g_hgan, g_infogan, 'G loss')
    
    plt.tight_layout()
    plt.savefig('{}/history_{}.svg'.format(save_dir, example))
    plt.savefig('{}/history_{}.eps'.format(save_dir, example))
    plt.close()
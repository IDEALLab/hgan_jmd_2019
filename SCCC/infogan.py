"""
InfoGAN

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
import tensorflow as tf

from shape_plot import plot_samples
from utils import preprocess, postprocess


EPSILON = 1e-7

class Model(object):
    
    def __init__(self, latent_dim, noise_dim, n_points, bezier_degree):
        
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        
        self.conc_points = sum(n_points)
        self.n_points = n_points
        
        self.bezier_degree = bezier_degree
        
    def encoder(self, x, name, reuse=tf.AUTO_REUSE, training=True):
        
        depth = 16
        kernel_size = (5,2)
        dropout = 0.4
        
        with tf.variable_scope(name, reuse=reuse):
        
            f = tf.layers.conv2d(x, depth*1, kernel_size, strides=(2,1), padding='same')
            f = tf.layers.batch_normalization(f, momentum=0.9)
            f = tf.nn.leaky_relu(f, alpha=0.2)
            f = tf.layers.dropout(f, dropout, training=training)
            
            f = tf.layers.conv2d(f, depth*2, kernel_size, strides=(2,1), padding='same')
            f = tf.layers.batch_normalization(f, momentum=0.9)
            f = tf.nn.leaky_relu(f, alpha=0.2)
            f = tf.layers.dropout(f, dropout, training=training)
            
            f = tf.layers.conv2d(f, depth*4, kernel_size, strides=(2,1), padding='same')
            f = tf.layers.batch_normalization(f, momentum=0.9)
            f = tf.nn.leaky_relu(f, alpha=0.2)
            f = tf.layers.dropout(f, dropout, training=training)
            
            f = tf.layers.conv2d(f, depth*8, kernel_size, strides=(2,1), padding='same')
            f = tf.layers.batch_normalization(f, momentum=0.9)
            f = tf.nn.leaky_relu(f, alpha=0.2)
            f = tf.layers.dropout(f, dropout, training=training)
            
            f = tf.layers.flatten(f)
            f = tf.layers.dense(f, 512)
            f = tf.layers.batch_normalization(f, momentum=0.9)
            f = tf.nn.leaky_relu(f, alpha=0.2)
        
            return f
        
    def bezier_net(self, inputs, n_points, bezier_degree, name, reuse=tf.AUTO_REUSE):
        
        depth_cpw = 32*8
        dim_cpw = int((bezier_degree+1)/8)
        kernel_size = (4,3)
#        noise_std = 0.01
        
        with tf.variable_scope(name, reuse=reuse):
        
            cpw = tf.layers.dense(inputs, 1024)
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
    
            cpw = tf.layers.dense(cpw, dim_cpw*3*depth_cpw)
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
            cpw = tf.reshape(cpw, (-1, dim_cpw, 3, depth_cpw))
    
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/2), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
    #        cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
            
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/4), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
    #        cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
            
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/8), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
    #        cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
            
            # Control points
            cp = tf.layers.conv2d(cpw, 1, (1,2), padding='valid') # batch_size x (bezier_degree+1) x 2 x 1
            cp = tf.nn.tanh(cp)
            cp = tf.squeeze(cp, axis=-1, name='control_point') # batch_size x (bezier_degree+1) x 2
            
            # Weights
            w = tf.layers.conv2d(cpw, 1, (1,3), padding='valid')
            w = tf.nn.sigmoid(w) # batch_size x (bezier_degree+1) x 1 x 1
            w = tf.squeeze(w, axis=-1, name='weight') # batch_size x (bezier_degree+1) x 1
            
            # Parameters at data points
            db = tf.layers.dense(inputs, 1024)
            db = tf.layers.batch_normalization(db, momentum=0.9)
            db = tf.nn.leaky_relu(db, alpha=0.2)
            
            db = tf.layers.dense(db, 256)
            db = tf.layers.batch_normalization(db, momentum=0.9)
            db = tf.nn.leaky_relu(db, alpha=0.2)
            
            db = tf.layers.dense(db, n_points-1)
            db = tf.nn.softmax(db) # batch_size x (n_data_points-1)
            
            ub = tf.pad(db, [[0,0],[1,0]], constant_values=0) # batch_size x n_data_points
            ub = tf.cumsum(ub, axis=1)
            ub = tf.minimum(ub, 1)
            ub = tf.expand_dims(ub, axis=-1) # 1 x n_data_points x 1
            
            # Bezier layer
            # Compute values of basis functions at data points
            num_control_points = bezier_degree + 1
            lbs = tf.tile(ub, [1, 1, num_control_points]) # batch_size x n_data_points x n_control_points
            pw1 = tf.range(0, num_control_points, dtype=tf.float32)
            pw1 = tf.reshape(pw1, [1, 1, -1]) # 1 x 1 x n_control_points
            pw2 = tf.reverse(pw1, axis=[-1])
            lbs = tf.add(tf.multiply(pw1, tf.log(lbs+EPSILON)), tf.multiply(pw2, tf.log(1-lbs+EPSILON))) # batch_size x n_data_points x n_control_points
            lc = tf.add(tf.lgamma(pw1+1), tf.lgamma(pw2+1))
            lc = tf.subtract(tf.lgamma(tf.cast(num_control_points, dtype=tf.float32)), lc) # 1 x 1 x n_control_points
            lbs = tf.add(lbs, lc) # batch_size x n_data_points x n_control_points
            bs = tf.exp(lbs)
            # Compute data points
            cp_w = tf.multiply(cp, w)
            dp = tf.matmul(bs, cp_w) # batch_size x n_data_points x 2
            bs_w = tf.matmul(bs, w) # batch_size x n_data_points x 1
            dp = tf.div(dp, bs_w) # batch_size x n_data_points x 2
            dp = tf.expand_dims(dp, axis=-1, name='x_fake') # batch_size x n_data_points x 2 x 1
            
            return dp, cp, w
    
    def circle_net(self, inputs, n_points, name, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope(name, reuse=reuse):
        
            x = tf.layers.dense(inputs, 1024)
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
    #        x += tf.random_normal(shape=tf.shape(x), stddev=noise_std)
    
            x = tf.layers.dense(x, 1024)
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
    #        x += tf.random_normal(shape=tf.shape(x), stddev=noise_std)
    
            x = tf.layers.dense(x, 128)
            x = tf.layers.batch_normalization(x, momentum=0.9)
            x = tf.nn.leaky_relu(x, alpha=0.2)
    #        x += tf.random_normal(shape=tf.shape(x), stddev=noise_std)
            
            # cx, cy, r
            cx = tf.layers.dense(x, 1, name='cx') # batch_size x 1
            cy = tf.layers.dense(x, 1, name='cy') # batch_size x 1
            r = tf.layers.dense(x, 1)
            r = tf.nn.softplus(r, name='r') # batch_size x 1
            
            # Parameters at data points
            ub = tf.lin_space(0.0, 2*np.pi, num=n_points)
            ub = tf.reshape(ub, [1, -1, 1])
            ub = tf.tile(ub, [tf.shape(x)[0], 1, 1]) # batch_size x n_data_points x 1
            
            # Circle layer
            dpx = tf.multiply(tf.expand_dims(r, axis=-1), tf.cos(ub))
            dpx = tf.add(tf.expand_dims(cx, axis=-1), dpx)
            dpy = tf.multiply(tf.expand_dims(r, axis=-1), tf.sin(ub))
            dpy = tf.add(tf.expand_dims(cy, axis=-1), dpy)
            dp = tf.concat([dpx, dpy], axis=-1) # batch_size x n_data_points x 2
            dp = tf.expand_dims(dp, axis=-1, name='x_fake') # batch_size x n_data_points x 2 x 1
            
            return dp
        
    def generator(self, c, z, name, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope(name, reuse=reuse):
            
            cz = tf.concat([c, z], axis=-1)
            dp0, cp, w = self.bezier_net(cz, self.n_points[0], self.bezier_degree[0], 'x0')
            dp1 = self.circle_net(cz, self.n_points[1], 'x1')
            dp2 = self.circle_net(cz, self.n_points[2], 'x2')
            dp3 = self.circle_net(cz, self.n_points[3], 'x3')
            
            return dp0, cp, w, dp1, dp2, dp3

    def discriminator(self, x0, x1, x2, x3, reuse=tf.AUTO_REUSE, training=True):
        
        with tf.variable_scope('D', reuse=reuse):
        
            f0 = self.encoder(x0, 'E0d', training=training)
            f1 = self.encoder(x1, 'E1d', training=training)
            f2 = self.encoder(x2, 'E2d', training=training)
            f3 = self.encoder(x3, 'E3d', training=training)
            f = tf.concat([f0, f1, f2, f3], axis=-1)
            
            c = tf.layers.dense(f, 1024)
            c = tf.layers.batch_normalization(c, momentum=0.9)
            c = tf.nn.leaky_relu(c, alpha=0.2)
            c = tf.layers.dense(c, 128)
            c = tf.layers.batch_normalization(c, momentum=0.9)
            c = tf.nn.leaky_relu(c, alpha=0.2)
            c_mean = tf.layers.dense(c, self.latent_dim, name='c_mean')
            c_logstd = tf.layers.dense(c, self.latent_dim, name='c_logstd')
            # Reshape to batch_size x 1 x child_latent_dim
            c_mean = tf.reshape(c_mean, (-1, 1, self.latent_dim))
            c_logstd = tf.reshape(c_logstd, (-1, 1, self.latent_dim))
            c = tf.concat([c_mean, c_logstd], axis=1, name='c_pred')
            
            d = tf.layers.dense(f, 1024)
            d = tf.layers.batch_normalization(d, momentum=0.9)
            d = tf.nn.leaky_relu(d, alpha=0.2)
            
            d = tf.layers.dense(d, 128)
            d = tf.layers.batch_normalization(d, momentum=0.9)
            d = tf.nn.leaky_relu(d, alpha=0.2)
            
            d = tf.layers.dense(d, 1)
            d = tf.add(d, 0, name='d_pred')
            
            return d, c

    def train(self, X_train, train_steps=2000, batch_size=256, save_interval=0, save_dir=None):
            
        assert X_train.shape[1] == self.conc_points
        
        X_train = preprocess(X_train)
        self.X0_train = X_train[:, :self.n_points[0]]
        self.X1_train = X_train[:, self.n_points[0]:self.n_points[0]+self.n_points[1]]
        self.X2_train = X_train[:, self.n_points[0]+self.n_points[1]:-self.n_points[3]]
        self.X3_train = X_train[:, -self.n_points[3]:]
        
        # Inputs
        self.x0 = tf.placeholder(tf.float32, shape=[None, self.n_points[0], 2, 1], name='x0')
        self.x1 = tf.placeholder(tf.float32, shape=[None, self.n_points[1], 2, 1], name='x1')
        self.x2 = tf.placeholder(tf.float32, shape=[None, self.n_points[2], 2, 1], name='x2')
        self.x3 = tf.placeholder(tf.float32, shape=[None, self.n_points[3], 2, 1], name='x3')
        self.c = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name='c')
        self.z = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name='z')
        
        # Targets
        c_target = tf.placeholder(tf.float32, shape=[None, self.latent_dim])
        
        # Outputs
        d_real, _ = self.discriminator(self.x0, self.x1, self.x2, self.x3)
        
        self.x0_fake, cp_train, w_train, self.x1_fake, self.x2_fake, self.x3_fake = self.generator(self.c, self.z, 'G')
        d_fake, self.c_fake_train = self.discriminator(self.x0_fake, self.x1_fake, self.x2_fake, self.x3_fake)
        
        self.d_test, self.c_test = self.discriminator(self.x0, self.x1, self.x2, self.x3, training=False)
        
        # Losses
        # Cross entropy losses for D
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        # Cross entropy losses for G
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        # Regularization for w, cp, a, and b (parent)
        r_w_loss = tf.reduce_mean(w_train[:,1:-1], axis=[1,2])
        cp_dist = tf.norm(cp_train[:,1:]-cp_train[:,:-1], axis=-1)
        r_cp_loss = tf.reduce_mean(cp_dist, axis=-1)
        r_cp_loss1 = tf.reduce_max(cp_dist, axis=-1)
        ends = cp_train[:,0] - cp_train[:,-1]
        r_ends_loss = tf.norm(ends, axis=-1) + tf.maximum(0.0, -10*ends[:,1]) # the second term penalizes intersecting at tail
        r_loss = r_w_loss + r_cp_loss + 0*r_cp_loss1 + r_ends_loss
        r_loss = tf.reduce_mean(r_loss)
        # Gaussian loss for Q
        def gaussian_loss(c, c_target):
            c_mean = c[:, 0, :]
            c_logstd = c[:, 1, :]
            epsilon = (c_target - c_mean) / (tf.exp(c_logstd) + EPSILON)
            q_loss = (c_logstd + 0.5 * tf.square(epsilon))
            q_loss = tf.reduce_mean(q_loss)
            return q_loss
        q_loss = gaussian_loss(self.c_fake_train, c_target)
        
        # Optimizers
        d_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)
        
        # Generator variables
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
        # Discriminator variables
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        
        # Training operations
        d_train_real = d_optimizer.minimize(d_loss_real, var_list=dis_vars)
        d_train_fake = d_optimizer.minimize(d_loss_fake + 0.1*q_loss, var_list=dis_vars)
        g_train = g_optimizer.minimize(g_loss + r_loss + 0.1*q_loss, var_list=g_vars)
        
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        
        # Create summaries to monitor losses
        tf.summary.scalar('D_loss_for_real', d_loss_real)
        tf.summary.scalar('D_loss_for_fake', d_loss_fake)
        tf.summary.scalar('G_loss', g_loss)
        tf.summary.scalar('R_loss', r_loss)
        tf.summary.scalar('Q_loss', q_loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Start training
        self.sess = tf.Session()
        
        # Run the initializer
        self.sess.run(init)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('{}/logs'.format(save_dir), graph=self.sess.graph)
    
        for t in range(train_steps):
    
            # Disriminator update
            ind = np.random.choice(self.X0_train.shape[0], size=batch_size, replace=False)
            X0_real = self.X0_train[ind]
            X1_real = self.X1_train[ind]
            X2_real = self.X2_train[ind]
            X3_real = self.X3_train[ind]
            _, dlr = self.sess.run([d_train_real, d_loss_real], feed_dict={self.x0: X0_real, self.x1: X1_real, self.x2: X2_real, self.x3: X3_real})
            
            X_latent = np.random.uniform(size=(batch_size, self.latent_dim))
            X_noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
            X0_fake, X1_fake, X2_fake, X3_fake = self.sess.run([self.x0_fake, self.x1_fake, self.x2_fake, self.x3_fake], feed_dict={self.c: X_latent, self.z: X_noise})

            assert not (np.any(np.isnan(X0_fake)) or np.any(np.isnan(X1_fake)) or np.any(np.isnan(X2_fake)) or np.any(np.isnan(X3_fake)))
            
            _, dlf, qld = self.sess.run([d_train_fake, d_loss_fake, q_loss],
                                        feed_dict={self.x0_fake: X0_fake, self.x1_fake: X1_fake, self.x2_fake: X2_fake, self.x3_fake: X3_fake, c_target: X_latent})
            
            # Generator update
            X_latent = np.random.uniform(size=(batch_size, self.latent_dim))
            X_noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
            X0_fake, X1_fake, X2_fake, X3_fake = self.sess.run([self.x0_fake, self.x1_fake, self.x2_fake, self.x3_fake], feed_dict={self.c: X_latent, self.z: X_noise})

            _, gl, rl, qlg = self.sess.run([g_train, g_loss, r_loss, q_loss],
                                           feed_dict={self.c: X_latent, self.z: X_noise, c_target: X_latent})
            
            summary_str = self.sess.run(merged_summary_op, 
                                        feed_dict={self.x0: X0_real, self.x1: X1_real, self.x2: X2_real, self.x3: X3_real,
                                                   self.x0_fake: X0_fake, self.x1_fake: X1_fake, self.x2_fake: X2_fake, self.x3_fake: X3_fake, 
                                                   self.c: X_latent, self.z: X_noise, c_target: X_latent})
            
            summary_writer.add_summary(summary_str, t+1)
            
            # Show messages
            log_mesg = "%d: [D] real %f fake %f q %f" % (t+1, dlr, dlf, qld)
            log_mesg = "%s  [G] %f reg %f q %f" % (log_mesg, gl, rl, qlg)
            print(log_mesg)
            
            if save_interval>0 and (t+1)%save_interval==0:
                
                # Save the variables to disk.
                save_path = saver.save(self.sess, '{}/model'.format(save_dir))
                print('Model saved in path: %s' % save_path)
                print('Plotting results ...')
                assemblies_list = self.synthesize(25)
                plot_samples(None, assemblies_list, scatter=True, alpha=.7, fname='{}/assemblies'.format(save_dir))
                    
    def restore(self, save_dir=None):
        
        self.sess = tf.Session()
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph('{}/model.meta'.format(save_dir))
        saver.restore(self.sess, tf.train.latest_checkpoint(save_dir))
        
        # Access and create placeholders variables            
        graph = tf.get_default_graph()
        self.x0 = graph.get_tensor_by_name('x0:0')
        self.x1 = graph.get_tensor_by_name('x1:0')
        self.x2 = graph.get_tensor_by_name('x2:0')
        self.x3 = graph.get_tensor_by_name('x3:0')
        self.x0_fake = graph.get_tensor_by_name('G/x0/x_fake:0')
        self.x1_fake = graph.get_tensor_by_name('G/x1/x_fake:0')
        self.x2_fake = graph.get_tensor_by_name('G/x2/x_fake:0')
        self.x3_fake = graph.get_tensor_by_name('G/x3/x_fake:0')
        self.c = graph.get_tensor_by_name('c:0')
        self.z = graph.get_tensor_by_name('z:0')
        self.c_test = graph.get_tensor_by_name('D_2/c_pred:0')
        self.d_test = graph.get_tensor_by_name('D_2/d_pred:0')

    def synthesize(self, X_latent):
        if isinstance(X_latent, int):
            N = X_latent
            X_latent = np.random.uniform(size=(N, self.latent_dim))
            X_noise = np.random.normal(scale=0.5, size=(N, self.noise_dim))
        else:
            N = X_latent.shape[0]
            X_noise = np.zeros((N, self.noise_dim))
        X0, X1, X2, X3 = self.sess.run([self.x0_fake, self.x1_fake, self.x2_fake, self.x3_fake], feed_dict={self.c: X_latent, self.z: X_noise})
        return [postprocess(X0), postprocess(X1), postprocess(X2), postprocess(X3)]
    
    def synthesize_assemblies(self, X_latent):
        X0, X1, X2, X3 = self.synthesize(X_latent)
        assemblies = np.concatenate((X0, X1, X2, X3), axis=1)
        return postprocess(assemblies)
    
    def embed(self, X0, X1, X2, X3):
        X0 = preprocess(X0)
        X1 = preprocess(X1)
        X2 = preprocess(X2)
        X3 = preprocess(X3)
        X_latent = self.sess.run(self.c_test, feed_dict={self.x0: X0, self.x1: X1, self.x2: X2, self.x3: X3})
        return X_latent[:,0,:]
    
    def pred_proba(self, X0, X1, X2, X3):
        X0 = preprocess(X0)
        X1 = preprocess(X1)
        X2 = preprocess(X2)
        X3 = preprocess(X3)
        logits = self.sess.run(self.d_test, feed_dict={self.x0: X0, self.x1: X1, self.x2: X2, self.x3: X3}).flatten()
        return 1. / (1. + np.exp(-logits))

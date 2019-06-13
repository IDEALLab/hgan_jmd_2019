"""
Hierarchical GAN with mutual information losses

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
import tensorflow as tf

from shape_plot import plot_grid
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
        
    def bezier_net(self, inputs, n_points, bezier_degree):
        
        depth_cpw = 32*8
        dim_cpw = int((bezier_degree+1)/8)
        kernel_size = (4,3)
#        noise_std = 0.01
        
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
    
    def circle_net(self, inputs, n_points):
        
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
        
    def parent_generator(self, c, z, name, n_points, bezier_degree=None, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope(name, reuse=reuse):
            
            cz = tf.concat([c, z], axis=-1)
            if bezier_degree is not None:
                dp, cp, w = self.bezier_net(cz, n_points, bezier_degree)
                return dp, cp, w
            else:
                dp = self.circle_net(cz, n_points)
                return dp
        
    def child_generator(self, c, z, fp, name, n_points, bezier_degree=None, reuse=tf.AUTO_REUSE):
        
#        noise_std = 0.01
        
        with tf.variable_scope(name, reuse=reuse):
            
            cz = tf.concat([c, z], axis=-1)
                
            cz = tf.layers.dense(cz, 128)
            cz = tf.layers.batch_normalization(cz, momentum=0.9)
            cz = tf.nn.leaky_relu(cz, alpha=0.2)
            
            cz = tf.layers.dense(cz, 256)
            cz = tf.layers.batch_normalization(cz, momentum=0.9)
            cz = tf.nn.leaky_relu(cz, alpha=0.2)
            
            x = tf.concat([cz, fp], axis=-1)
            if bezier_degree is not None:
                dp, cp, w = self.bezier_net(x, n_points, bezier_degree)
                return dp, cp, w
            else:
                dp = self.circle_net(x, n_points)
                return dp

    def discriminator(self, x0, x1, x2, reuse=tf.AUTO_REUSE, training=True):
        
        with tf.variable_scope('D', reuse=reuse):
        
            f0 = self.encoder(x0, 'E0d', training=training)
            c0 = tf.layers.dense(f0, 128)
            c0 = tf.layers.batch_normalization(c0, momentum=0.9)
            c0 = tf.nn.leaky_relu(c0, alpha=0.2)
            c0_mean = tf.layers.dense(c0, self.latent_dim[0], name='c0_mean')
            c0_logstd = tf.layers.dense(c0, self.latent_dim[0], name='c0_logstd')
            # Reshape to batch_size x 1 x parent_latent_dim
            c0_mean = tf.reshape(c0_mean, (-1, 1, self.latent_dim[0]))
            c0_logstd = tf.reshape(c0_logstd, (-1, 1, self.latent_dim[0]))
            c0 = tf.concat([c0_mean, c0_logstd], axis=1, name='c0_pred')
            
            f1 = self.encoder(x1, 'E1d', training=training)
            f01 = tf.concat([f0, f1], axis=-1)
            c1 = tf.layers.dense(f01, 1024)
            c1 = tf.layers.batch_normalization(c1, momentum=0.9)
            c1 = tf.nn.leaky_relu(c1, alpha=0.2)
            c1 = tf.layers.dense(c1, 128)
            c1 = tf.layers.batch_normalization(c1, momentum=0.9)
            c1 = tf.nn.leaky_relu(c1, alpha=0.2)
            c1_mean = tf.layers.dense(c1, self.latent_dim[1], name='c1_mean')
            c1_logstd = tf.layers.dense(c1, self.latent_dim[1], name='c1_logstd')
            # Reshape to batch_size x 1 x child_latent_dim
            c1_mean = tf.reshape(c1_mean, (-1, 1, self.latent_dim[1]))
            c1_logstd = tf.reshape(c1_logstd, (-1, 1, self.latent_dim[1]))
            c1 = tf.concat([c1_mean, c1_logstd], axis=1, name='c1_pred')
            
            f2 = self.encoder(x2, 'E2d', training=training)
            f012 = tf.concat([f01, f2], axis=-1)
            c2 = tf.layers.dense(f012, 1024)
            c2 = tf.layers.batch_normalization(c2, momentum=0.9)
            c2 = tf.nn.leaky_relu(c2, alpha=0.2)
            c2 = tf.layers.dense(c2, 128)
            c2 = tf.layers.batch_normalization(c2, momentum=0.9)
            c2 = tf.nn.leaky_relu(c2, alpha=0.2)
            c2_mean = tf.layers.dense(c2, self.latent_dim[2], name='c2_mean')
            c2_logstd = tf.layers.dense(c2, self.latent_dim[2], name='c2_logstd')
            # Reshape to batch_size x 1 x child_latent_dim
            c2_mean = tf.reshape(c2_mean, (-1, 1, self.latent_dim[2]))
            c2_logstd = tf.reshape(c2_logstd, (-1, 1, self.latent_dim[2]))
            c2 = tf.concat([c2_mean, c2_logstd], axis=1, name='c2_pred')
            
            d = tf.layers.dense(f012, 1024)
            d = tf.layers.batch_normalization(d, momentum=0.9)
            d = tf.nn.leaky_relu(d, alpha=0.2)
            
            d = tf.layers.dense(d, 128)
            d = tf.layers.batch_normalization(d, momentum=0.9)
            d = tf.nn.leaky_relu(d, alpha=0.2)
            
            d = tf.layers.dense(d, 1)
            d = tf.add(d, 0, name='d_pred')
            
            return d, c0, c1, c2

    def train(self, X_train, train_steps=2000, batch_size=256, save_interval=0, save_dir=None):
            
        assert X_train.shape[1] == self.conc_points
        
        X_train = preprocess(X_train)
        self.X0_train = X_train[:, :self.n_points[0]]
        self.X1_train = X_train[:, self.n_points[0]:self.n_points[0]+self.n_points[1]]
        self.X2_train = X_train[:, -self.n_points[2]:]
        
        # Inputs
        self.x0 = tf.placeholder(tf.float32, shape=[None, self.n_points[0], 2, 1], name='x0')
        self.c0 = tf.placeholder(tf.float32, shape=[None, self.latent_dim[0]], name='c0')
        self.z0 = tf.placeholder(tf.float32, shape=[None, self.noise_dim[0]], name='z0')
        self.x1 = tf.placeholder(tf.float32, shape=[None, self.n_points[1], 2, 1], name='x1')
        self.c1 = tf.placeholder(tf.float32, shape=[None, self.latent_dim[1]], name='c1')
        self.z1 = tf.placeholder(tf.float32, shape=[None, self.noise_dim[1]], name='z1')
        self.x2 = tf.placeholder(tf.float32, shape=[None, self.n_points[2], 2, 1], name='x2')
        self.c2 = tf.placeholder(tf.float32, shape=[None, self.latent_dim[2]], name='c2')
        self.z2 = tf.placeholder(tf.float32, shape=[None, self.noise_dim[2]], name='z2')
        
        # Targets
        c0_target = tf.placeholder(tf.float32, shape=[None, self.latent_dim[0]])
        c1_target = tf.placeholder(tf.float32, shape=[None, self.latent_dim[1]])
        c2_target = tf.placeholder(tf.float32, shape=[None, self.latent_dim[2]])
        
        # Outputs
        d_real, _, _, _ = self.discriminator(self.x0, self.x1, self.x2)
        
        self.x0_fake, cp_train, w_train = self.parent_generator(self.c0, self.z0, 'G0', self.n_points[0], self.bezier_degree[0])
        f0 = self.encoder(self.x0_fake, 'E0', training=False)
        self.x1_fake = self.child_generator(self.c1, self.z1, f0, 'G1', self.n_points[1])
        f1 = self.encoder(self.x1_fake, 'E1', training=False)
        self.x2_fake = self.child_generator(self.c2, self.z2, tf.concat([f0, f1], axis=-1), 'G2', self.n_points[2])
        d_fake, self.c0_fake_train, self.c1_fake_train, self.c2_fake_train = self.discriminator(self.x0_fake, self.x1_fake, self.x2_fake)
        
        self.d_test, self.c0_test, self.c1_test, self.c2_test = self.discriminator(self.x0, self.x1, self.x2, training=False)
        
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
        q_loss = gaussian_loss(self.c0_fake_train, c0_target) + \
                 gaussian_loss(self.c1_fake_train, c1_target) + \
                 gaussian_loss(self.c2_fake_train, c2_target)
        
        # Optimizers
        d_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)
        
        # Generator variables
        g0_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G0')
        g1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G1')
        g2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G2')
        # Discriminator variables
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        
        # Training operations
        d_train_real = d_optimizer.minimize(d_loss_real, var_list=dis_vars)
        d_train_fake = d_optimizer.minimize(d_loss_fake + 0.1*q_loss, var_list=dis_vars)
        g_train = g_optimizer.minimize(g_loss + r_loss + 0.1*q_loss, var_list=[g0_vars, g1_vars, g2_vars])
        
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
            _, dlr = self.sess.run([d_train_real, d_loss_real], feed_dict={self.x0: X0_real, self.x1: X1_real, self.x2: X2_real})
            
            X0_latent = np.random.uniform(size=(batch_size, self.latent_dim[0]))
            X0_noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim[0]))
            X0_fake = self.sess.run(self.x0_fake, feed_dict={self.c0: X0_latent, self.z0: X0_noise})
            X1_latent = np.random.uniform(size=(batch_size, self.latent_dim[1]))
            X1_noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim[1]))
            X1_fake = self.sess.run(self.x1_fake, feed_dict={self.c1: X1_latent, self.z1: X1_noise, self.x0_fake: X0_fake})
            X2_latent = np.random.uniform(size=(batch_size, self.latent_dim[2]))
            X2_noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim[2]))
            X2_fake = self.sess.run(self.x2_fake, feed_dict={self.c2: X2_latent, self.z2: X2_noise, self.x0_fake: X0_fake, self.x1_fake: X1_fake})

            assert not (np.any(np.isnan(X0_fake)) or np.any(np.isnan(X1_fake)) or np.any(np.isnan(X2_fake)))
            
            _, dlf, qld = self.sess.run([d_train_fake, d_loss_fake, q_loss],
                                        feed_dict={self.x0_fake: X0_fake, self.x1_fake: X1_fake, self.x2_fake: X2_fake,
                                                   c0_target: X0_latent, c1_target: X1_latent, c2_target: X2_latent})
            
            # Generator update
            X0_latent = np.random.uniform(size=(batch_size, self.latent_dim[0]))
            X0_noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim[0]))
            X0_fake = self.sess.run(self.x0_fake, feed_dict={self.c0: X0_latent, self.z0: X0_noise})
            X1_latent = np.random.uniform(size=(batch_size, self.latent_dim[1]))
            X1_noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim[1]))
            X1_fake = self.sess.run(self.x1_fake, feed_dict={self.c1: X1_latent, self.z1: X1_noise, self.x0_fake: X0_fake})
            X2_latent = np.random.uniform(size=(batch_size, self.latent_dim[2]))
            X2_noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim[2]))
            X2_fake = self.sess.run(self.x2_fake, feed_dict={self.c2: X2_latent, self.z2: X2_noise, self.x0_fake: X0_fake, self.x1_fake: X1_fake})

            _, gl, rl, qlg = self.sess.run([g_train, g_loss, r_loss, q_loss],
                                           feed_dict={self.c0: X0_latent, self.z0: X0_noise, 
                                                      self.c1: X1_latent, self.z1: X1_noise,
                                                      self.c2: X2_latent, self.z2: X2_noise,
                                                      c0_target: X0_latent, c1_target: X1_latent, c2_target: X2_latent})
            
            summary_str = self.sess.run(merged_summary_op, 
                                        feed_dict={self.x0: X0_real, self.x1: X1_real, self.x2: X2_real,
                                                   self.x0_fake: X0_fake, self.x1_fake: X1_fake, self.x2_fake: X2_fake, 
                                                   self.c0: X0_latent, self.z0: X0_noise, c0_target: X0_latent, 
                                                   self.c1: X1_latent, self.z1: X1_noise, c1_target: X1_latent, 
                                                   self.c2: X2_latent, self.z2: X2_noise, c2_target: X2_latent,})
            
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
                plot_grid(5, gen_func=self.synthesize_x0, d=self.latent_dim[0], 
                          scale=.95, scatter=True, alpha=.7, fname='{}/x0'.format(save_dir))
                plot_grid(5, gen_func=self.synthesize_x1, d=self.latent_dim[1], 
                          scale=.95, scatter=True, alpha=.7, fname='{}/x1'.format(save_dir))
                plot_grid(5, gen_func=self.synthesize_x2, proba_func=self.pred_proba, d=self.latent_dim[2], 
                          scale=.95, scatter=True, alpha=.7, fname='{}/x2'.format(save_dir))
                    
    def restore(self, save_dir=None):
        
        self.sess = tf.Session()
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph('{}/model.meta'.format(save_dir))
        saver.restore(self.sess, tf.train.latest_checkpoint(save_dir))
        
        # Access and create placeholders variables            
        graph = tf.get_default_graph()
        self.x0 = graph.get_tensor_by_name('x0:0')
        self.c0 = graph.get_tensor_by_name('c0:0')
        self.z0 = graph.get_tensor_by_name('z0:0')
        self.x0_fake = graph.get_tensor_by_name('G0/x_fake:0')
        self.x1 = graph.get_tensor_by_name('x1:0')
        self.c1 = graph.get_tensor_by_name('c1:0')
        self.z1 = graph.get_tensor_by_name('z1:0')
        self.x1_fake = graph.get_tensor_by_name('G1/x_fake:0')
        self.x2 = graph.get_tensor_by_name('x2:0')
        self.c2 = graph.get_tensor_by_name('c2:0')
        self.z2 = graph.get_tensor_by_name('z2:0')
        self.x2_fake = graph.get_tensor_by_name('G2/x_fake:0')
        self.c0_test = graph.get_tensor_by_name('D_2/c0_pred:0')
        self.c1_test = graph.get_tensor_by_name('D_2/c1_pred:0')
        self.c2_test = graph.get_tensor_by_name('D_2/c2_pred:0')
        self.d_test = graph.get_tensor_by_name('D_2/d_pred:0')

    def synthesize_x0(self, X0_latent):
        if isinstance(X0_latent, int):
            N = X0_latent
            X0_latent = np.random.uniform(size=(N, self.latent_dim[0]))
            X0_noise = np.random.normal(scale=0.5, size=(N, self.noise_dim[0]))
        else:
            N = X0_latent.shape[0]
            X0_noise = np.zeros((N, self.noise_dim[0]))
        X0 = self.sess.run(self.x0_fake, feed_dict={self.c0: X0_latent, self.z0: X0_noise})
        return [postprocess(X0)]
    
    def synthesize_x1(self, X1_latent, parents=None):
        if isinstance(X1_latent, int):
            N = X1_latent
            X1_latent = np.random.uniform(size=(N, self.latent_dim[1]))
            X1_noise = np.random.normal(scale=0.5, size=(N, self.noise_dim[1]))
        else:
            N = X1_latent.shape[0]
            X1_noise = np.zeros((N, self.noise_dim[1]))
        if parents is None:
            X0 = self.synthesize_x0(1)[0]
        else:
            X0 = parents[0]
        X0 = preprocess(X0)
        X0 = np.tile(X0, (N,1,1,1))
        X1 = self.sess.run(self.x1_fake, feed_dict={self.c1: X1_latent, self.z1: X1_noise, self.x0_fake: X0})
        return [postprocess(X1), postprocess(X0)]
    
    def synthesize_x2(self, X2_latent, parents=None):
        if isinstance(X2_latent, int):
            N = X2_latent
            X2_latent = np.random.uniform(size=(N, self.latent_dim[2]))
            X2_noise = np.random.normal(scale=0.5, size=(N, self.noise_dim[2]))
        else:
            N = X2_latent.shape[0]
            X2_noise = np.zeros((N, self.noise_dim[2]))
        if parents is None:
            X1, X0 = self.synthesize_x1(1)
        else:
            X0, X1 = parents
        X0 = preprocess(X0)
        X0 = np.tile(X0, (N,1,1,1))
        X1 = preprocess(X1)
        X1 = np.tile(X1, (N,1,1,1))
        X2 = self.sess.run(self.x2_fake, feed_dict={self.c2: X2_latent, self.z2: X2_noise, self.x0_fake: X0, self.x1_fake: X1})
        return [postprocess(X2), postprocess(X0), postprocess(X1)]
    
    def synthesize_assemblies(self, N):
        X0_latent = np.random.uniform(size=(N, self.latent_dim[0]))
        X0_noise = np.random.normal(scale=0.5, size=(N, self.noise_dim[0]))
        X0 = self.sess.run(self.x0_fake, feed_dict={self.c0: X0_latent, self.z0: X0_noise})
        X1_latent = np.random.uniform(size=(N, self.latent_dim[1]))
        X1_noise = np.random.normal(scale=0.5, size=(N, self.noise_dim[1]))
        X1 = self.sess.run(self.x1_fake, feed_dict={self.c1: X1_latent, self.z1: X1_noise, 
                                                    self.x0_fake: X0})
        X2_latent = np.random.uniform(size=(N, self.latent_dim[2]))
        X2_noise = np.random.normal(scale=0.5, size=(N, self.noise_dim[2]))
        X2 = self.sess.run(self.x2_fake, feed_dict={self.c2: X2_latent, self.z2: X2_noise, 
                                                    self.x0_fake: X0, self.x1_fake: X1})
        assemblies = np.concatenate((X0, X1, X2), axis=1)
        return postprocess(assemblies)
    
    def embed(self, X0, X1, X2):
        X0 = preprocess(X0)
        X1 = preprocess(X1)
        X2 = preprocess(X2)
        X0_latent, X1_latent, X2_latent = self.sess.run([self.c0_test, self.c1_test, self.c2_test], 
                                                        feed_dict={self.x0: X0, self.x1: X1, self.x2: X2})
        return X0_latent[:,0,:], X1_latent[:,0,:], X2_latent[:,0,:]
    
    def pred_proba(self, X0, X1, X2):
        X0 = preprocess(X0)
        X1 = preprocess(X1)
        X2 = preprocess(X2)
        logits = self.sess.run(self.d_test, feed_dict={self.x0: X0, self.x1: X1, self.x2: X2}).flatten()
        return 1. / (1. + np.exp(-logits))

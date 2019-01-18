# coding:utf-8
'''
    Double-AutoEncoder + Latent Adversarial + Joint Classifier
'''

import matplotlib
matplotlib.use("Agg")
import tensorflow as tf 
import numpy as np
import os
from generator import CNN_Generator
from classifier import MLP_Classifier
from discriminator import CNN_Latent_discriminator
from utils import BasicTrainFramework
from datamanager import datamanager_mnist
from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

mnist_path = "data/mnist/mnist.npz"
colorbackground_mnist_path = "data/mnist/mnist_colorback.npz"
colornumeral_mnist_path = "data/mnist/mnist_colornum.npz"

class DAEGAN(BasicTrainFramework):
    def __init__(self, batch_size, version='daegan'):
        super(DAEGAN, self).__init__(batch_size, version)

        self.critic_iter = 3

        self.data_A = datamanager_mnist(datapath=mnist_path, train_ratio=0.8, fold_k=None, expand_dim=True, norm=True, seed=0)
        self.data_B = datamanager_mnist(datapath=colorbackground_mnist_path, train_ratio=0.8, fold_k=None, expand_dim=False, norm=True, seed=1)

        self.sample_A = self.data_A(self.batch_size, phase='test', var_list=['data', 'labels'])
        self.sample_B = self.data_B(self.batch_size, phase='test', var_list=['data', 'labels'])

        self.autoencoder_A = CNN_Generator(output_dim=1, name='ae_A')
        self.autoencoder_B = CNN_Generator(output_dim=3, name='ae_B')

        self.Latent_classifier = MLP_Classifier(output_dim=10, layers=[128]*2, name='latent_c')
        self.Latent_discriminator = CNN_Latent_discriminator(name='latent_d')

        self.build_placeholder()
        self.build_network()
        self.build_optimizer()
        self.build_summary()

        self.build_sess()
        self.build_dirs()

    def build_placeholder(self):
        self.source_A = tf.placeholder(tf.float32, (self.batch_size, 28, 28, 1))
        self.source_B = tf.placeholder(tf.float32, (self.batch_size, 28, 28, 3))
        self.label_A = tf.placeholder(tf.float32, (self.batch_size, 10))
        self.label_B = tf.placeholder(tf.float32, (self.batch_size, 10))
        
    def build_network(self):
        # source_A->emb_A->cyc_A, emb_A->cls_A, emb_A->logit_A
        self.cyc_A, self.emb_A = self.autoencoder_A(self.source_A)
        self.cls_A = self.Latent_classifier(tf.reshape(self.emb_A, [self.batch_size, -1]))
        self.latent_logit_A = self.Latent_discriminator(self.emb_A)
        # source_B->emb_B->cyc_B, emb_B->cls_B, emb_B->logit_B
        self.cyc_B, self.emb_B = self.autoencoder_B(self.source_B)
        self.cls_B = self.Latent_classifier(tf.reshape(self.emb_B, [self.batch_size, -1]), reuse=True)
        self.latent_logit_B = self.Latent_discriminator(self.emb_B, reuse=True)
        # source_A->emb_A->cyc_AtoB, source_B->emb_B->cyc_BtoA
        self.cyc_AtoB, _ = self.autoencoder_B(self.emb_A, decode_mode=True, is_training=False, reuse=True)
        self.cyc_BtoA, _ = self.autoencoder_A(self.emb_B, decode_mode=True, is_training=False, reuse=True)

    def build_optimizer(self):
        # latent GAN
        self.latent_D_loss_A = tf.reduce_mean(tf.squared_difference(self.latent_logit_B, 1.0))
        self.latent_D_loss_B = tf.reduce_mean(tf.square(self.latent_logit_A))
        self.latent_D_loss = self.latent_D_loss_A + self.latent_D_loss_B
        self.latent_G_loss = tf.reduce_mean(tf.squared_difference(self.latent_logit_A, 1.0))
        # latent classifier
        self.C_loss_A = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.cls_A, labels=self.label_A))
        self.C_loss_B = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.cls_B, labels=self.label_B))
        self.batch_acc_A = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.nn.softmax(self.cls_A), axis=1), tf.argmax(self.label_A, axis=1))))
        self.batch_acc_B = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.nn.softmax(self.cls_B), axis=1), tf.argmax(self.label_B, axis=1))))
        # double-autoencoder
        self.cyc_loss_A = tf.reduce_mean(tf.squared_difference(self.source_A, self.cyc_A))
        self.cyc_loss_B = tf.reduce_mean(tf.squared_difference(self.source_B, self.cyc_B))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.latent_D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.latent_D_loss, var_list=self.Latent_discriminator.vars)
            self.latent_G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.latent_G_loss, var_list=self.autoencoder_A.vars)

            self.C_solver_A = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.C_loss_A, var_list=self.autoencoder_A.vars + self.Latent_classifier.vars)
            self.C_solver_B = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.C_loss_B, var_list=self.autoencoder_B.vars + self.Latent_classifier.vars)

            self.cyc_solver_A = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.cyc_loss_A, var_list=self.autoencoder_A.vars)
            self.cyc_solver_B = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.cyc_loss_B, var_list=self.autoencoder_B.vars)

    def build_summary(self):
        acc_A_sum = tf.summary.scalar('A_acc', self.batch_acc_A)
        acc_B_sum = tf.summary.scalar('B_acc', self.batch_acc_B)
        ae_A_sum = tf.summary.scalar('A_aeloss', self.cyc_loss_A)
        ae_B_sum = tf.summary.scalar('B_aeloss', self.cyc_loss_B)
        g_sum = tf.summary.scalar('g_loss', self.latent_G_loss)
        d_sum = tf.summary.scalar('d_loss', self.latent_D_loss)
        self.summary = tf.summary.merge([acc_A_sum,acc_B_sum,ae_A_sum,ae_B_sum,g_sum,d_sum])
    
    def sample(self, epoch=0):
        def plot_density(mat, color, label):
            mat = mat[mat>0]
            p, _ = np.histogram(mat, bins=np.linspace(0, 15, 500), density=True)
            plt.plot(np.linspace(0, 15, len(p)), p, color=color, label=label, linewidth=2)
        
        emb_A = []
        for _ in range(100):
            A = self.data_A(self.batch_size, var_list=['data'])
            emb_A.append(self.sess.run(self.emb_A, feed_dict={self.source_A:A['data']}))
        plot_density(np.concatenate(emb_A, 0), 'r', 'emb_A')
        del emb_A 
        
        emb_B = []
        for _ in range(100):
            B = self.data_B(self.batch_size, var_list=['data'])
            emb_B.append(self.sess.run(self.emb_B, feed_dict={self.source_B:B['data']}))
        plot_density(np.concatenate(emb_B, 0), 'b', 'emb_B')
        del emb_B 

        plt.legend()
        plt.savefig(os.path.join(self.fig_dir, "hist_{}.png".format(epoch)))
        plt.clf()

        cyc_A = self.sess.run(self.cyc_A, feed_dict={self.source_A: self.sample_A['data']})
        cyc_A = np.concatenate([np.concatenate([seq for seq in cyc_A[i*5:(i+1)*5]], 1) for i in range(5)], axis=0)
        plt.imshow(cyc_A[:,:,0], cmap=plt.cm.gray)
        plt.savefig(os.path.join(self.fig_dir, "cycA_{}.png".format(epoch)))
        plt.clf()

        cyc_B = self.sess.run(self.cyc_B, feed_dict={self.source_B: self.sample_B['data']})
        cyc_B = np.concatenate([np.concatenate([seq for seq in cyc_B[i*5:(i+1)*5]], 1) for i in range(5)], axis=0)
        plt.imshow(cyc_B)
        plt.savefig(os.path.join(self.fig_dir, "cycB_{}.png".format(epoch)))
        plt.clf()

        cross_AtoB = self.sess.run(self.cyc_AtoB, feed_dict={self.source_A: self.sample_A['data']})
        cross_AtoB = np.concatenate([np.concatenate([seq for seq in cross_AtoB[i*5:(i+1)*5]], 1) for i in range(5)], axis=0)
        plt.imshow(cross_AtoB)
        plt.savefig(os.path.join(self.fig_dir, "cross_AtoB_{}.png".format(epoch)))
        plt.clf()

        cross_BtoA = self.sess.run(self.cyc_BtoA, feed_dict={self.source_B: self.sample_B['data']})
        cross_BtoA = np.concatenate([np.concatenate([seq for seq in cross_BtoA[i*5:(i+1)*5]], 1) for i in range(5)], axis=0)
        plt.imshow(cross_BtoA[:,:,0], cmap=plt.cm.gray)
        plt.savefig(os.path.join(self.fig_dir, "cross_BtoA_{}.png".format(epoch)))
        plt.clf()

    def train(self, epoches=1):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        batches_per_epoch = self.data_A.train_num // self.batch_size

        for epoch in range(epoches):
            self.data_A.shuffle_train(seed=epoch)
            self.data_B.shuffle_train(seed=epoch+1)

            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx 

                A = self.data_A(self.batch_size, var_list=['data', 'labels'])
                B = self.data_B(self.batch_size, var_list=['data', 'labels'])

                feed_dict = {self.source_A:A['data'], self.source_B:B['data'], 
                             self.label_A:A['labels'], self.label_B:B['labels']}

                self.sess.run([self.cyc_solver_A, self.cyc_solver_B], feed_dict=feed_dict)
                self.sess.run([self.C_solver_A, self.C_solver_B], feed_dict=feed_dict)

                for _ in range(self.critic_iter):
                    self.sess.run(self.latent_D_solver, feed_dict=feed_dict)
                self.sess.run(self.latent_G_solver, feed_dict=feed_dict)

                if cnt % 10 == 0:
                    cyca, cycb, acca, accb, gl, dl, sum_str = self.sess.run([self.cyc_loss_A, self.cyc_loss_B, self.batch_acc_A, self.batch_acc_B, self.latent_G_loss, self.latent_D_loss, self.summary], feed_dict=feed_dict)
                    print self.version+" epoch [%3d/%3d] iter [%3d/%3d] cyca=%.3f cycb=%.3f acca=%.3f accb=%.3f gl=%.3f dl=%.3f" \
                        % (epoch, epoches, idx, batches_per_epoch, cyca, cycb, acca, accb, gl, dl)
                    self.writer.add_summary(sum_str, cnt)
            if epoch % 2 == 0:
                self.sample(epoch)
        self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)




daegan = DAEGAN(64)
daegan.train(20)
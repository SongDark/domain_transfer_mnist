# coding:utf-8
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf 
import numpy as np
import os
from generator import CNN_Generator
from discriminator import CNN_Discriminator
from utils import BasicTrainFramework, event_reader
from datamanager import datamanager_mnist
from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

mnist_path = "data/mnist/mnist.npz"
colorbackground_mnist_path = "data/mnist/mnist_colorback.npz"
colornumeral_mnist_path = "data/mnist/mnist_colornum.npz"

class CycleGAN(BasicTrainFramework):
    def __init__(self, data_A, data_B, batch_size, gan_type='gan', version="cyclegan"):
        self.gan_type = gan_type
        super(CycleGAN, self).__init__(batch_size, version)

        self.data_A = data_A 
        self.data_B = data_B 
        self.sample_data_A = self.data_A(self.batch_size, phase='test', var_list=['data'])
        self.sample_data_B = self.data_B(self.batch_size, phase='test', var_list=['data'])

        self.critic_iter = 3

        self.generator_AtoB = CNN_Generator(output_dim=3, name="cnn_generator_AtoB")
        self.generator_BtoA = CNN_Generator(output_dim=1, name="cnn_generator_BtoA")
        self.discriminator_A = CNN_Discriminator(name="cnn_discriminator_A")
        self.discriminator_B = CNN_Discriminator(name="cnn_discriminator_B")

        self.build_placeholder()
        self.build_network()
        self.build_optimizer()
        self.build_summary()

        self.build_sess()
        self.build_dirs()
    
    def build_placeholder(self):
        # gray image
        self.source_A = tf.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)
        # colored image
        self.source_B = tf.placeholder(shape=(self.batch_size, 28, 28, 3), dtype=tf.float32)

    def build_network(self):
        # cyclegan
        self.fake_B = self.generator_AtoB(self.source_A, is_training=True, reuse=False)
        self.fake_A = self.generator_BtoA(self.source_B, is_training=True, reuse=False)
        self.fake_B_test = self.generator_AtoB(self.source_A, is_training=False, reuse=True)
        self.fake_A_test = self.generator_BtoA(self.source_B, is_training=False, reuse=True)

        self.logit_real_A, _ = self.discriminator_A(self.source_A, is_training=True, reuse=False)
        self.logit_real_B, _ = self.discriminator_B(self.source_B, is_training=True, reuse=False)
        self.logit_fake_A, _ = self.discriminator_A(self.fake_A, is_training=True, reuse=True)
        self.logit_fake_B, _ = self.discriminator_B(self.fake_B, is_training=True, reuse=True)

        self.cyc_A = self.generator_BtoA(self.fake_B, is_training=True, reuse=True)
        self.cyc_B = self.generator_AtoB(self.fake_A, is_training=True, reuse=True)

    def build_optimizer(self):
        # self.reconstruct_loss = mse(self.fake_A, self.source_A, self.batch_size)
        # if self.gan_type == 'gan':
        #     self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_real, labels=tf.ones_like(self.logit_real)))
        #     self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.zeros_like(self.logit_fake)))
        #     self.D_loss = self.D_loss_real + self.D_loss_fake
        #     self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.ones_like(self.logit_fake)))
        # elif self.gan_type == 'wgan':
        #     self.D_loss_real = -tf.reduce_mean(self.logit_real)
        #     self.D_loss_fake = tf.reduce_mean(self.logit_fake)
        #     self.D_loss = self.D_loss_real + self.D_loss_fake
        #     self.G_loss = - self.D_loss_fake
        #     self.D_clip = [v.assign(tf.clip_by_value(v, -0.1, 0.1)) for v in self.discriminator_B.vars]
        self.D_loss_real_A = tf.reduce_mean(tf.squared_difference(self.logit_real_A, 1))
        self.D_loss_real_B = tf.reduce_mean(tf.squared_difference(self.logit_real_B, 1))
        self.D_loss_fake_A = tf.reduce_mean(tf.square(self.logit_fake_A))
        self.D_loss_fake_B = tf.reduce_mean(tf.square(self.logit_fake_B))
        self.D_loss_A = self.D_loss_real_A + self.D_loss_fake_A
        self.D_loss_B = self.D_loss_real_B + self.D_loss_fake_B 
        self.reconstruct_loss = tf.reduce_mean(tf.abs(self.source_A-self.cyc_A)) \
            + tf.reduce_mean(tf.abs(self.source_B-self.cyc_B))

        self.G_loss_A = tf.reduce_mean(tf.squared_difference(self.logit_fake_B, 1)) + 10*self.reconstruct_loss
        self.G_loss_B = tf.reduce_mean(tf.squared_difference(self.logit_fake_A, 1)) + 10*self.reconstruct_loss
        
        self.D_solver_A = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss_A, var_list=self.discriminator_A.vars)
        self.D_solver_B = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss_B, var_list=self.discriminator_B.vars)
        self.G_solver_A = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.G_loss_A, var_list=self.generator_AtoB.vars)
        self.G_solver_B = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.G_loss_B, var_list=self.generator_BtoA.vars)

    def build_summary(self):
        R_sum = tf.summary.scalar('reconstruct_loss', self.reconstruct_loss)
        D_sum_A = tf.summary.scalar('D_loss_A', self.D_loss_A)
        G_sum_A = tf.summary.scalar('G_loss_A', self.G_loss_A)
        D_sum_B = tf.summary.scalar('D_loss_B', self.D_loss_B)
        G_sum_B = tf.summary.scalar('G_loss_B', self.G_loss_B)
        self.summary = tf.summary.merge([R_sum, D_sum_A, G_sum_A, D_sum_B, G_sum_B])
    
    def plot(self, imgs, savepath):
        # imgs [bz, 28, 28, 3 or 1]
        tmp = [[] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                tmp[i].append(imgs[i*5+j])
            tmp[i] = np.concatenate(tmp[i], 1)
        tmp = np.concatenate(tmp, 0)
        if tmp.shape[-1] == 1:
            plt.imshow(tmp[:,:,0], cmap=plt.cm.gray)
        else:
            plt.imshow(tmp)
        plt.savefig(savepath)
        plt.clf()

    def sample(self, epoch):
        print "sample at epoch {}".format(epoch)
        feed_dict = {self.source_A : self.sample_data_A['data']}
        G = self.sess.run(self.fake_B_test, feed_dict=feed_dict)
        self.plot(G, os.path.join(self.fig_dir, "AtoB_epoch_{}.png".format(epoch)))

        feed_dict = {self.source_B : self.sample_data_B['data']}
        G = self.sess.run(self.fake_A_test, feed_dict=feed_dict)
        self.plot(G, os.path.join(self.fig_dir, "BtoA_epoch_{}.png".format(epoch)))

        if epoch == 0:
            self.plot(self.sample_data_A['data'], os.path.join(self.fig_dir, "ori_A.png"))
            self.plot(self.sample_data_B['data'], os.path.join(self.fig_dir, "ori_B.png"))

    def plot_loss(self):
        res = event_reader(self.log_dir, 
            names=['reconstruct_loss', 'G_loss_A', 'G_loss_B', 'D_loss_A', 'D_loss_B'])
        total_iters = len(res['reconstruct_loss'])
        plt.clf()
        plt.gca().set_ylim([0, 1])
        plt.plot(range(total_iters), res['reconstruct_loss'], label='reconstruct_loss')
        plt.legend()
        plt.savefig(os.path.join(self.fig_dir, "reconstruct_loss.png"))
        plt.clf()

        plt.gca().set_ylim([0, 9])
        plt.plot(range(total_iters), res['G_loss_A'], label='G_loss_A')
        plt.plot(range(total_iters), res['D_loss_A'], label='D_loss_A')
        plt.legend()
        plt.savefig(os.path.join(self.fig_dir, "GD_loss_A.png"))
        plt.clf()

        plt.gca().set_ylim([0, 9])
        plt.plot(range(total_iters), res['G_loss_B'], label='G_loss_B')
        plt.plot(range(total_iters), res['D_loss_B'], label='D_loss_B')
        plt.legend()
        plt.savefig(os.path.join(self.fig_dir, "GD_loss_B.png"))
        plt.clf()

    
    def train(self, epoches=1):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        batches_per_epoch = self.data_A.train_num // self.batch_size

        for epoch in range(epoches):
            self.data_A.shuffle_train(seed=epoch)
            self.data_B.shuffle_train(seed=epoch+1)

            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx 

                A = self.data_A(self.batch_size, var_list=['data'])
                B = self.data_B(self.batch_size, var_list=['data'])

                feed_dict = {
                    self.source_A : A['data'], 
                    self.source_B : B['data']
                }

                # # update G
                # self.sess.run(self.G_solver_A, feed_dict=feed_dict)
                # # update D
                # self.sess.run(self.D_solver_B, feed_dict=feed_dict)
                # # update G
                # self.sess.run(self.G_solver_B, feed_dict=feed_dict)
                # # update D
                # self.sess.run(self.D_solver_A, feed_dict=feed_dict)
                self.sess.run([self.G_solver_A, self.D_solver_B], feed_dict=feed_dict)
                self.sess.run([self.G_solver_B, self.D_solver_A], feed_dict=feed_dict)

                if cnt % 10 == 0:
                    da, db, ga, gb, r, sum_str = self.sess.run([self.D_loss_A, self.D_loss_B, self.G_loss_A, self.G_loss_B, self.reconstruct_loss, self.summary], feed_dict=feed_dict)
                    print self.version + " Epoch [%3d/%3d] Iter [%3d/%3d] Da=%.3f Db=%.3f Ga=%.3f Gb=%.3f R=%.3f" % (epoch, epoches, idx, batches_per_epoch, da, db, ga, gb, r)
                    self.writer.add_summary(sum_str, cnt)
            if epoch % 20 == 0:
                self.sample(epoch)
        self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)


data_A = datamanager_mnist(datapath=mnist_path, train_ratio=0.8, fold_k=None, expand_dim=True, norm=True, seed=0)
data_B = datamanager_mnist(datapath=colorbackground_mnist_path, train_ratio=0.8, fold_k=None, expand_dim=False, norm=True, seed=1)
cyclegan = CycleGAN(data_A, data_B, 64, gan_type='', version='cyclegan')
cyclegan.train(100)

cyclegan.plot_loss()


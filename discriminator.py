import tensorflow as tf 
from utils import *

# class CNN_Discriminator(BasicBlock):
#     def __init__(self, class_num=None, name=None):
#         super(CNN_Discriminator, self).__init__(None, name or "Discriminator_CNN")
#         self.class_num = class_num
    
#     def __call__(self, x, y=None, sn=False, is_training=True, reuse=False):
#         with tf.variable_scope(self.name, reuse=reuse):
#             batch_size = x.get_shape().as_list()[0]
#             if y is not None:
#                 ydim = y.get_shape().as_list()[-1]
#                 y = tf.reshape(y, [batch_size, 1, 1, ydim])
#                 x = conv_cond_concat(x, y) # [bz, 28, 28, 11]
#             # [bz, 14, 14, 64]
#             net = lrelu(conv2d(x, 64, 4, 4, 2, 2, sn=sn, padding="SAME", name='d_conv1'), name='d_l1')
#             # [bz, 7, 7, 128]
#             net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, sn=sn, padding="SAME", name='d_conv2'), is_training, name='d_bn2'), name='d_l2')
#             net = tf.reshape(net, [batch_size, 7*7*128])
#             # [bz, 1024]
#             net = lrelu(bn(dense(net, 1024, sn=sn, name='d_fc3'), is_training, name='d_bn3'), name='d_l3')
#             # [bz, 1]
#             yd = dense(net, 1, sn=sn, name='D_dense')
#             if self.class_num:
#                 yc = dense(net, self.class_num, sn=sn, name='C_dense')
#                 return yd, net, yc 
#             else:
#                 return yd, net

class CNN_Discriminator(BasicBlock):
    def __init__(self, class_num=None, name=None):
        super(CNN_Discriminator, self).__init__(None, name or "Discriminator_CNN")
        self.class_num = class_num
    
    def __call__(self, x, y=None, sn=False, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            batch_size = x.get_shape().as_list()[0]
            if y is not None:
                ydim = y.get_shape().as_list()[-1]
                y = tf.reshape(y, [batch_size, 1, 1, ydim])
                x = conv_cond_concat(x, y) # [bz, 28, 28, 11]
            # [bz, 14, 14, 64]
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, sn=sn, padding="SAME", name='d_conv1'), name='d_l1')
            # [bz, 7, 7, 128]
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, sn=sn, padding="SAME", name='d_conv2'), is_training, name='d_bn2'), name='d_l2')
            # [bz, 4, 4, 256]
            net = lrelu(bn(conv2d(net, 256, 4, 4, 2, 2, sn=sn, padding="SAME", name='d_conv3'), is_training, name='d_bn3'), name='d_l3')
            # [bz, 4, 4, 512]
            net = lrelu(bn(conv2d(net, 512, 4, 4, 1, 1, sn=sn, padding="SAME", name='d_conv4'), is_training, name='d_bn4'), name='d_l4')
            # [bz, 4, 4, 1]
            yd = conv2d(net, 1, 4, 4, 1, 1, sn=sn, padding="SAME", name='d_conv5')

            return yd, net

class CNN_Latent_discriminator(BasicBlock):
    def __init__(self, name=None):
        super(CNN_Latent_discriminator, self).__init__(None, name or "CNN_Latent_discriminator")
    
    def __call__(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):

            c1 = lrelu(bn(conv2d(x, 64, 3, 3, 2, 2, padding='SAME', name='d_c1'), is_training, name='d_b1'), name='d_l1')
            c2 = conv2d(c1, 1, 3, 3, 2, 2, padding='SAME', name='d_c2') # [bz, 4, 4, 1]

        return c2
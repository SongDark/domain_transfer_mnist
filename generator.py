import tensorflow as tf 
from utils import * 

# class CNN_Generator(BasicBlock):
#     def __init__(self, output_dim, name=None):
#         super(CNN_Generator, self).__init__(None, name or "CNN_Generator")
#         self.output_dim = output_dim

#     def __call__(self, x, sn=False, is_training=True, reuse=False):
#         batch_size = x.get_shape().as_list()[0]
#         with tf.variable_scope(self.name, reuse=reuse):
#             net = lrelu(conv2d(x, 64, 4, 4, 2, 2, sn=sn, padding="SAME", name='g_conv1'), name="g_l1")
            
#             net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, sn=sn, padding="SAME", name='g_conv2'), is_training, name="g_bn2"), name="g_l2")

#             net = tf.reshape(net, [batch_size, 7*7*128])

#             net = lrelu(bn(dense(net, 1024, sn=sn, name='g_fc3'), is_training, name='g_bn3'), name='g_l3')

#             net = tf.nn.relu(bn(dense(net, 7*7*128, sn=sn, name='g_fc4'), is_training, name='g_bn4'), name='g_l4')

#             net = tf.reshape(net, [batch_size, 7, 7, 128])

#             net = tf.nn.relu(bn(deconv2d(net, 64, 4, 4, 2, 2, padding="SAME", name='g_deconv5'), is_training, name='g_bn5'))

#             out = tf.nn.sigmoid(deconv2d(net, self.output_dim, 4, 4, 2, 2, padding="SAME", name='g_deconv6'))

#         return out

class CNN_Generator(BasicBlock):
    def __init__(self, output_dim, name=None):
        super(CNN_Generator, self).__init__(None, name or "CNN_Generator")
        self.output_dim = output_dim

    def __call__(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            pad_x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
            c1 = tf.nn.relu(bn(conv2d(pad_x, 32, 7, 7, 1, 1, padding="VALID", name="g_c1"), is_training, name='g_bn1'))
            c2 = tf.nn.relu(bn(conv2d(c1, 64, 3, 3, 2, 2, padding="SAME", name="g_c2"), is_training, name='g_bn2'))
            c3 = tf.nn.relu(bn(conv2d(c2, 128, 3, 3, 2, 2, padding="SAME", name='g_c3'), is_training, name='g_bn3'))

            r1 = resnet_block(c3, 128, is_training, name='r1')
            r2 = resnet_block(r1, 128, is_training, name='r2') 
            r3 = resnet_block(r2, 128, is_training, name='r3') 
            r4 = resnet_block(r3, 128, is_training, name='r4')
            r5 = resnet_block(r4, 128, is_training, name='r5')
            r6 = resnet_block(r5, 128, is_training, name='r6')

            d1 = tf.nn.relu(bn(deconv2d(r6, 64, 3, 3, 2, 2, padding="SAME", name='g_dc1'), is_training, name='g_bn4'))
            d2 = tf.nn.relu(bn(deconv2d(d1, 32, 3, 3, 2, 2, padding="SAME", name='g_dc2'), is_training, name='g_bn5'))
            d2_pad = tf.pad(d2, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
            c4 = bn(conv2d(d2_pad, self.output_dim, 7, 7, 1, 1, padding="VALID", name="g_c4"), is_training, name='g_bn6')
            
            out = tf.nn.sigmoid(c4)
        return out


# G = CNN_Generator(output_dim=3)
# x = tf.random_normal(shape=(5,28,28,1), dtype=tf.float32)
# y = G(x)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(y).shape
import tensorflow as tf 
from utils import * 

# class MLP_Classifier(BasicBlock):
#     def __init__(self, class_num, name=None):
#         super(MLP_Classifier, self).__init__(None, name or 'MLP_Classifier')
#         self.class_num = class_num 
    
#     def __call__(self, x, is_training=True, reuse=False):
#         with tf.variable_scope(self.name, reuse=reuse):
#             batch_size = x.get_shape().as_list()[0]
#             x = tf.reshape(x, [batch_size, -1])
#             net = lrelu(bn(dense(x, 128, name='c_fc1'), is_training, name='c_bn1'), name='c_l1')
#             out_logit = dense(net, self.class_num, name='c_fc2')
#             out_softmax = tf.nn.softmax(out_logit)
#         return out_logit, out_softmax

class MLP_Classifier(BasicBlock):
    def __init__(self, output_dim, layers, name=None):
        super(MLP_Classifier, self).__init__(None, name or 'mlp')
        self.output_dim = output_dim
        self.layers = layers

    def __call__(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            net = x 
            for i in range(len(self.layers)):
                net = dense(net, self.layers[i], name='fc{}'.format(i))
                net = tf.nn.relu(net)
            net = dense(net, self.output_dim, name='mlp_out')
        
        return net 
# -*- coding:utf-8 -*-
# Created Time: Thu 13 Apr 2017 04:07:50 PM CST
# $Author: Taihong Xiao <xiaotaihong@126.com>

from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
from dataset import config, Dataset
from six.moves import reduce


class Model(object):
    def __init__(self, is_train=True, nhwc=config.nhwc, max_iter=config.max_iter, weight_decay=config.weight_decay,
                 second_ratio=config.second_ratio):
        super(Model, self).__init__()
        self.is_train = is_train
        self.batch_size, self.height, self.width, self.channel = nhwc
        self.max_iter = max_iter
        self.g_lr = tf.placeholder(tf.float32)
        self.d_lr = tf.placeholder(tf.float32)
        self.weight_decay = weight_decay
        self.second_ratio = second_ratio
        self.reuse = {}
        self.build_model()

    def leakyRelu(self, x, alpha=0.2):
        return tf.maximum(alpha * x, x)

    def make_conv(self, name, X, shape, strides):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            return tf.nn.conv2d(X, W, strides=strides, padding='SAME')

    def make_conv_bn(self, name, X, shape, strides):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            return tf.layers.batch_normalization(
                tf.nn.conv2d(X, W, strides=strides, padding='SAME'),
                training=self.is_train
            )

    def make_fc(self, name, X, out_dim):
        in_dim = X.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=[in_dim, out_dim],
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            b = tf.get_variable('b',
                                shape=[out_dim],
                                initializer=tf.zeros_initializer(),
                                )
            return tf.add(tf.matmul(X, W), b)

    def make_fc_bn(self, name, X, out_dim):
        in_dim = X.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=[in_dim, out_dim],
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            b = tf.get_variable('b',
                                shape=[out_dim],
                                initializer=tf.zeros_initializer(),
                                )
            X = tf.add(tf.matmul(X, W), b)
            return tf.layers.batch_normalization(X, training=self.is_train)

    def make_deconv(self, name, X, filter_shape, out_shape, strides):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=filter_shape,
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            return tf.nn.conv2d_transpose(X, W, output_shape=out_shape, strides=strides, padding='SAME')

    def make_deconv_bn(self, name, X, filter_shape, out_shape, strides):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=filter_shape,
                                initializer=tf.random_normal_initializer(stddev=0.02),
                                )
            return tf.layers.batch_normalization(
                tf.nn.conv2d_transpose(X, W,
                                       output_shape=out_shape, strides=strides, padding='SAME'
                                       ), training=self.is_train
            )

    def discriminator(self, name, image, inputsdims = 512, expand=False):
        # X = image / 255.0
        X = image
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            if expand:#expand use for?
                X = X / 255.0
                X = self.make_conv('conv1', X, shape=[4, 4, 3, 128], strides=[1, 2, 2, 1])
                X = self.leakyRelu(X, 0.2)
                # print(name, X.get_shape())

                X = self.make_conv_bn('conv2', X, shape=[4, 4, 128, 256], strides=[1, 2, 2, 1])
                X = self.leakyRelu(X, 0.2)
                # print(name, X.get_shape())

                X = self.make_conv_bn('conv3', X, shape=[4, 4, 256, 512], strides=[1, 2, 2, 1])
                X = self.leakyRelu(X, 0.2)
                # print(name, X.get_shape())

            X = self.make_conv_bn('conv4', X, shape=[4, 4, inputsdims, 512], strides=[1, 2, 2, 1])
            X = self.leakyRelu(X, 0.2)

            flat_dim = reduce(lambda x, y: x * y, X.get_shape().as_list()[1:])
            X = tf.reshape(X, [-1, flat_dim])
            X = self.make_fc('fct', X, 1)
            # X = tf.nn.sigmoid(X)
            return X

    def splitter(self, name, image):
        X = image / 255.0
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            X = self.make_conv('conv1', X, shape=[4, 4, 3, 128], strides=[1, 2, 2, 1])
            X = self.leakyRelu(X, 0.2)

            X = self.make_conv_bn('conv2', X, shape=[4, 4, 128, 256], strides=[1, 2, 2, 1])
            X = self.leakyRelu(X, 0.2)

            X = self.make_conv_bn('conv3', X, shape=[4, 4, 256, 512], strides=[1, 2, 2, 1])
            X = self.leakyRelu(X, 0.2)

            num_ch = int(X.get_shape().as_list()[-1] * self.second_ratio)
            return X[:, :, :, :-num_ch], X[:, :, :, -num_ch:]

    def joiner(self, name, A, x):
        X = tf.concat([A, x], axis=-1)
        # X0 = X
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            X = self.make_deconv_bn('deconv1', X, filter_shape=[4, 4, 512, 512],
                                    out_shape=[self.batch_size, int(self.height / 4), int(self.width / 4), 512],
                                    strides=[1, 2, 2, 1])
            X = tf.nn.relu(X)

            X = self.make_deconv_bn('deconv2', X, filter_shape=[4, 4, 256, 512],
                                    out_shape=[self.batch_size, int(self.height / 2), int(self.width / 2), 256],
                                    strides=[1, 2, 2, 1])
            X = tf.nn.relu(X)

            X = self.make_deconv('deconv3', X, filter_shape=[4, 4, self.channel, 256],
                                 out_shape=[self.batch_size, self.height, self.width, self.channel],
                                 strides=[1, 2, 2, 1])
            b = tf.get_variable('b', shape=[1, 1, 1, self.channel], initializer=tf.zeros_initializer())
            X = X + b

            X = (tf.tanh(X) + 1) * 255.0 / 2
            return X

    def build_model(self):
        self.Ax = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, self.channel], name='data1')
        self.Ae_dst = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, self.channel], name='data1_dst')
        self.Be = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, self.channel], name='data2')

        self.A, self.x = self.splitter('G_splitter', self.Ax)
        self.B, self.e = self.splitter('G_splitter', self.Be)

        self.Ax2 = self.joiner('G_joiner', self.A, self.x)
        self.Be2 = self.joiner('G_joiner', self.B, tf.zeros_like(self.e))
        # self.Be2 = self.joiner('G_joiner', self.B, self.e)

        # crossover
        self.Bx = self.joiner('G_joiner', self.B, self.x)
        self.Ae = self.joiner('G_joiner', self.A, tf.zeros_like(self.e))
        # self.Ae = self.joiner('G_joiner', self.A, self.e)

        self.real_Ax = self.discriminator('D_Ax', self.Ax - self.Ae_dst, expand=True)
        self.fake_Bx = self.discriminator('D_Ax', self.Bx - self.Be, expand=True)

        self.real_Be = self.discriminator('D_Be', self.Be - self.Be, expand=True)
        self.fake_Ae = self.discriminator('D_Be', self.Ae - self.Ae_dst, expand=True)

        # self.A1, self.x1 = self.splitter('D_splitter', self.Ax)
        # self.B1, self.e1 = self.splitter('D_splitter', self.Be)
        # self.Ax_ = tf.concat([self.Ax[self.batch_size // 2:], self.Ax[:self.batch_size // 2]], axis=0)
        # self.A1_, self.x1_ = self.splitter('D_splitter', self.Ax_)
        # self.Be_ = tf.concat([self.Be[self.batch_size // 2:], self.Be[:self.batch_size // 2]], axis=0)
        # self.B1_, self.e1_ = self.splitter('D_splitter', self.Be_)
        # self.B2, self.x2 = self.splitter('D_splitter', self.Bx)
        # self.A2, self.e2 = self.splitter('D_splitter', self.Ae)
        #
        #
        # self.real_x = self.discriminator('D_x', self.x1, int(512*self.second_ratio))
        # self.fake_x = self.discriminator('D_x', self.x2, int(512*self.second_ratio))
        # self.real_e = self.discriminator('D_e', self.e1, int(512*self.second_ratio))
        # self.fake_e = self.discriminator('D_e', self.e2, int(512*self.second_ratio))
        # self.real_eB = self.discriminator('D_B', self.B1, int(512*(1-self.second_ratio)))
        # self.fake_xB = self.discriminator('D_B', self.B2, int(512*(1-self.second_ratio)))
        # self.real_eC = self.discriminator('D_A', self.B1, int(512*(1-self.second_ratio)))
        # self.fake_eA = self.discriminator('D_A', self.A2, int(512*(1-self.second_ratio)))
        #
        # self.real_x_ = self.discriminator('D_x', self.x1_, int(512 * self.second_ratio))
        # # self.fake_x_ = self.discriminator('D_x', self.x2, int(512 * self.second_ratio))
        # self.real_e_ = self.discriminator('D_e', self.e1_, int(512 * self.second_ratio))
        # # self.fake_e_ = self.discriminator('D_e', self.e2, int(512 * self.second_ratio))
        # self.real_eB_ = self.discriminator('D_B', self.B1_, int(512 * (1 - self.second_ratio)))
        # # self.fake_xB_ = self.discriminator('D_B', self.B2, int(512 * (1 - self.second_ratio)))
        # self.real_eC_ = self.discriminator('D_A', self.B1_, int(512 * (1 - self.second_ratio)))
        # # self.fake_eA_ = self.discriminator('D_A', self.A2, int(512 * (1 - self.second_ratio)))

        # self.real_A = self.discriminator('D_AB', self.A, 128)
        # self.fake_A = self.discriminator('D_AB', self.A2, 128)
        # self.fake_A2 = self.discriminator('D_AB', self.A3, 128)
        # self.real_B = self.discriminator('D_AB', self.B, 128)
        # self.fake_B = self.discriminator('D_AB', self.B2, 128)
        # self.fake_B2 = self.discriminator('D_AB', self.B3, 128)

        # variable list
        self.g_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_joiner') \
                          + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_splitter')


        self.d_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_Ax')\
                          + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_Be')
                          # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_A') \
                          # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_B') \
                          # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_x') \
                          # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_e')\
                          # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_splitter')
                          # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_splitter_e')
                          # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_A') \
                          # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_Ax') \
                          # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_Be')

        # ratio_whole = 1.0
        # G loss
        self.G_loss = {}
        self.G_loss['null_e'] = tf.reduce_mean(tf.abs(self.e))
        # self.G_loss['max_x'] = -tf.reduce_mean(tf.nn.moments(self.x, [1,2,3])[1])
        self.G_loss['cycle_Ax'] = tf.reduce_mean(tf.abs(self.Ax - self.Ax2)) / 255.0
        self.G_loss['cycle_Be'] = tf.reduce_mean(tf.abs(self.Be - self.Be2)) / 255.0
        self.G_loss['GAN'] = -(tf.reduce_mean(self.fake_Bx)
                               + tf.reduce_mean(self.fake_Ae)
                               #  tf.reduce_mean(self.fake_x)
                               # + tf.reduce_mean(self.fake_e)
                               # # + tf.reduce_mean(self.fake_xB)
                               # + tf.reduce_mean(self.fake_eA)
                               # # + tf.reduce_mean(self.fake_B2)
                               # # + tf.reduce_mean(self.fake_A2)
                               )
        # self.G_loss['GAN_whole'] = -(tf.reduce_mean(self.fake_Bx) + tf.reduce_mean(self.fake_Ae)) * ratio_whole
        self.G_loss['parallelogram'] = 0.01 * tf.reduce_mean(tf.abs(self.Ax + self.Be - self.Bx - self.Ae))
        # self.G_loss['parallelogram'] = 0.01 * tf.reduce_mean(tf.abs(self.Ax - self.Bx) + tf.abs(self.Be - self.Ae))
        self.G_loss['addition1'] = 0.01 * tf.reduce_mean(tf.abs(tf.nn.moments(self.Ae, [1, 2])[1]))
        self.loss_G_nodecay = sum(self.G_loss.values())

        self.loss_G_decay = 0.0
        for w in self.g_var_list:
            if w.name.startswith('G') and w.name.endswith('W:0'):
                self.loss_G_decay += 0.5 * self.weight_decay * tf.reduce_mean(tf.square(w))
                # print(w.name)

        self.loss_G = self.loss_G_decay + self.loss_G_nodecay

        # D loss
        self.D_loss = {}
        # self.D_loss['null_e'] = tf.reduce_mean(tf.abs(self.e2))
        # self.D_loss['x'] = tf.reduce_mean(self.fake_x - self.real_x)
        # self.D_loss['e'] = tf.reduce_mean(self.fake_e - self.real_e)
        # self.D_loss['A'] = tf.reduce_mean(self.fake_eA - self.real_eC)
        # self.D_loss['B'] = tf.reduce_mean(self.fake_xB - self.real_eB)
        # self.D_loss['x2'] = tf.reduce_mean(self.fake_x - self.real_x_)
        # self.D_loss['e2'] = tf.reduce_mean(self.fake_e - self.real_e_)
        # self.D_loss['A2'] = tf.reduce_mean(self.fake_eA - self.real_eC_)
        # self.D_loss['B2'] = tf.reduce_mean(self.fake_xB - self.real_eB_)
        # self.D_loss['DG'] = (tf.reduce_mean(tf.abs(self.A1 - self.A)) * (1-self.second_ratio)
        #                      + tf.reduce_mean(tf.abs(self.x1 - self.x)) * self.second_ratio
        #                      + tf.reduce_mean(tf.abs(self.B1 - self.B)) * (1-self.second_ratio)
        #                      + tf.reduce_mean(tf.abs(self.e1 - self.e)) * self.second_ratio
        #                      )
        # self.D_loss['A_B'] = tf.reduce_mean(self.fake_A - self.real_B) * ratio_s
        # self.D_loss['B_A'] = tf.reduce_mean(self.fake_B - self.real_A) * ratio_s

        # self.D_loss['whole'] = (tf.reduce_mean(self.fake_Bx - self.real_Ax) + tf.reduce_mean(self.fake_Ae - self.real_Be)) * ratio_whole
        self.D_loss['x'] = tf.reduce_mean(self.fake_Bx - self.real_Ax)
        self.D_loss['e'] = tf.reduce_mean(self.fake_Ae - self.real_Be)
        self.loss_D = sum(self.D_loss.values())

        # G, D optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.g_opt = tf.train.RMSPropOptimizer(self.g_lr, decay=0.8).minimize(self.loss_G, var_list=self.g_var_list)
            self.d_opt = tf.train.RMSPropOptimizer(self.d_lr, decay=0.8).minimize(self.loss_D, var_list=self.d_var_list)

        # clip weights in D
        with tf.name_scope('clip_d'):
            self.clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_var_list]


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    celebA = Dataset('Eyeglasses')
    image_batch = celebA.input()

    GeneGAN = Model()



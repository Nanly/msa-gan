import os
import tensorflow as tf
import numpy as np
from scipy import misc



class GenerateImg():
    def __init__(self, GeneGAN, model_dir):
        self.batchs = 64
        self.Model = GeneGAN(is_train=True, nhwc=[self.batchs, 64, 64, 3])

        with open('/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/datasets/try_1/plane64.txt', 'r', encoding='utf-8') as file:
        #with open('mysdir/datasets/try_1/plane64.txt', 'r', encoding='utf-8') as file:
            plane = [i.rstrip() for i in file.readlines()]
        with open('/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/datasets/try_1/bkgd64.txt', 'r', encoding='utf-8') as file:
        #with open('mysdir/datasets/try_1/bkgd64.txt', 'r', encoding='utf-8') as file:
            bkgd = [i.rstrip() for i in file.readlines()]
        self.Ax = np.concatenate([np.expand_dims(misc.imresize(misc.imread(i), (self.Model.height, self.Model.width)), axis=0) for i in plane],
                                     axis=0).astype('float32')
        self.Be = np.concatenate([np.expand_dims(misc.imresize(misc.imread(i), (self.Model.height, self.Model.width)), axis=0) for i in bkgd],
                                     axis=0).astype('float32')
        if len(self.Ax)+len(self.Be) != 64+64:
            print('No 64 imgs')

        # create graph and restore
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        # save_variables = tf.trainable_variables()
        # g_list = tf.global_variables()
        # bn_variables = [g for g in g_list if 'moving_mean' in g.name] + [g for g in g_list if
        #                                                                  'moving_variance' in g.name]
        # save_variables += bn_variables
        saver = tf.train.Saver()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, model_dir)

    def convert_paths2img(self, paths):
        return np.concatenate([np.expand_dims(misc.imresize(misc.imread(i), (self.Model.height, self.Model.width)), axis=0) for i in paths],
                              axis=0).astype('float32')

    def _expand_batchimg(self, Ax, Be):
        if Ax.ndim != 4 and Be.ndim != 4:
            print('ndims erros')
            return -1
        if Ax.shape != (self.batchs, 64, 64, 3):
            Ax = np.concatenate([Ax, self.Ax[:self.batchs-len(Ax)]])
        if Be.shape != (self.batchs, 64, 64, 3):
            Be = np.concatenate([Be, self.Be[:self.batchs-len(Be)]])
        return Ax, Be

    def swap_attribute(self, input_src, input_attr, out_dir=None, lens=1):
        src_img = np.array(input_src)
        att_img = np.array(input_attr)
        src_img, att_img = self._expand_batchimg(src_img, att_img)
        if not out_dir:
            out_dir = ['/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/test_pics/exp_default1/generated/out1.jpg',
                       '/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/test_pics/exp_default1/generated/out2.jpg']
            #out_dir = ['mysdir/test_pics/exp_default1/generated/out1.jpg',
            #           'mysdir/test_pics/exp_default1/generated/out2.jpg']

        out2, out1 = self.sess.run([self.Model.Ae, self.Model.Bx], feed_dict={self.Model.Ax: src_img, self.Model.Be: att_img})
        # misc.imsave(out_dir[0], out1[0].astype('uint8'))
        # misc.imsave(out_dir[1], out2[0].astype('uint8'))
        return out1[:lens], out2[:lens]

    def interpolation(self, input_src, input_attr, inter_num=5, out_dir=None):
        src_img = np.array(input_src)
        att_img = np.array(input_attr)
        src_img, att_img = self._expand_batchimg(src_img, att_img)
        if not out_dir:
            out_dir = ['/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/test_pics/exp_default1/generated/out1.jpg',
                       '/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/test_pics/exp_default1/generated/out2.jpg']
            #out_dir = ['mysdir/test_pics/exp_default1/generated/out1.jpg',
            #           'mysdir/test_pics/exp_default1/generated/out2.jpg']

        out = src_img[0]
        for i in range(1, inter_num + 1):
            lambda_i = i / float(inter_num)
            self.Model.out_i = self.Model.joiner('G_joiner', self.Model.B, self.Model.x * lambda_i)
            out_i = self.sess.run(self.Model.out_i, feed_dict={self.Model.Ax: att_img, self.Model.Be: src_img})
            out = np.concatenate((out, out_i[0]), axis=1)
        misc.imsave(out_dir, out.astype('uint8'))

    def interpolation_matrix_mAx(self, input_src, input_attr, size, out_dir=None):
        def convert2src64(data, datas):
            data_ = data[np.newaxis, :]
            return np.concatenate([data_, datas[1:]], axis=0)

        src_img = np.array(input_src)
        att_imgs = np.array(input_attr)
        src_img, att_imgs = self._expand_batchimg(src_img, att_imgs)
        if not out_dir:
            out_dir = ['/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/test_pics/exp_default1/generated/four.jpg']
            #out_dir = ['mysdir/test_pics/exp_default1/generated/four.jpg']

        m, n = size
        h, w = self.Model.height, self.Model.width

        rows = [[1 - i / float(m - 1), i / float(m - 1)] for i in range(m)]
        cols = [[1 - i / float(n - 1), i / float(n - 1)] for i in range(n)]
        four_tuple = []
        for row in rows:
            for col in cols:
                four_tuple.append([row[0] * col[0], row[0] * col[1], row[1] * col[0], row[1] * col[1]])

        attributes_ = self.sess.run(self.Model.B, feed_dict={self.Model.Be: att_imgs})
        attributes = attributes_[:4]
        x = self.sess.run(self.Model.x, feed_dict={self.Model.Ax: src_img})
        # B = B_[:1]

        cnt = 0
        out = np.zeros((0, w * n, self.Model.channel))
        for i in range(m):
            out_row = np.zeros((h, 0, self.Model.channel))
            for j in range(n):
                four = four_tuple[cnt]
                attribute = sum([four[i] * attributes[i] for i in range(4)])
                # print(attribute.shape)
                img = self.sess.run(self.Model.joiner('G_joiner', convert2src64(attribute, attributes_), x))[0]
                out_row = np.concatenate((out_row, img.astype('float64')), axis=1)
                cnt += 1
            out = np.concatenate((out, out_row), axis=0)

        first_col = np.concatenate((att_imgs[0], 255 * np.ones(((m - 2) * h, w, 3)), att_imgs[2]), axis=0)

        last_col = np.concatenate((att_imgs[1], 255 * np.ones(((m - 2) * h, w, 3)), att_imgs[3]), axis=0)

        out_canvas = np.concatenate((first_col, out, last_col), axis=1)
        # misc.imsave(out_dir[0], out_canvas)
        return out_canvas

    def interpolation_matrix_mBe(self, input_src, input_attr, size, out_dir=None):
        def convert2src64(data, datas):
            data_ = data[np.newaxis, :]
            return np.concatenate([data_, datas[1:]], axis=0)

        src_img = np.array(input_src)
        att_imgs = np.array(input_attr)
        src_img, att_imgs = self._expand_batchimg(src_img, att_imgs)
        if not out_dir:
            out_dir = ['/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/test_pics/exp_default1/generated/four.jpg']
            #out_dir = ['mysdir/test_pics/exp_default1/generated/four.jpg']

        m, n = size
        h, w = self.Model.height, self.Model.width

        rows = [[1 - i / float(m - 1), i / float(m - 1)] for i in range(m)]
        cols = [[1 - i / float(n - 1), i / float(n - 1)] for i in range(n)]
        four_tuple = []
        for row in rows:
            for col in cols:
                four_tuple.append([row[0] * col[0], row[0] * col[1], row[1] * col[0], row[1] * col[1]])

        attributes_ = self.sess.run(self.Model.B, feed_dict={self.Model.Be: att_imgs})
        attributes = attributes_[:4]
        x = self.sess.run(self.Model.x, feed_dict={self.Model.Ax: src_img})

        cnt = 0
        out = np.zeros((0, w * n, self.Model.channel))
        for i in range(m):
            out_row = np.zeros((h, 0, self.Model.channel))
            for j in range(n):
                four = four_tuple[cnt]
                attribute = sum([four[i] * attributes[i] for i in range(4)])
                # print(attribute.shape)
                img = self.sess.run(self.Model.joiner('G_joiner', convert2src64(attribute, attributes_), x))[0]
                out_row = np.concatenate((out_row, img.astype('float64')), axis=1)
                cnt += 1
            out = np.concatenate((out, out_row), axis=0)

        first_col = np.concatenate((att_imgs[0], 255 * np.ones(((m - 2) * h, w, 3)), att_imgs[2]), axis=0)

        last_col = np.concatenate((att_imgs[1], 255 * np.ones(((m - 2) * h, w, 3)), att_imgs[3]), axis=0)

        out_canvas = np.concatenate((first_col, out, last_col), axis=1)
        # misc.imsave(os.path.join(out_dir, 'four_matrix.jpg'), out_canvas)
        return out_canvas

def merge_imgs(imgs, sizes):
    x, y = sizes

    cont = 0
    out = np.zeros((0, 64, 3), dtype='uint8')
    for i in range(x):
        out_row = np.zeros((0, 64, 3), dtype='uint8')
        for j in range(y):
            out_row = np.concatenate((out_row, imgs[cont].astype('uint8')), axis=1)
            cont += 1
        out = np.concatenate((out, out_row), axis=0)
    return out

def imsave_type32(img, path):
    misc.imsave(path, img.astype('uint8'))

def imsave_type8(img, path):
    misc.imsave(path, img)


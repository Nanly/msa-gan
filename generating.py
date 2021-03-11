import os

"""
os.chdir('/home/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master')
from mysdir.utils import *
from mysdir.trains2.plane2.sub_input.model import Model
import matplotlib.pyplot as plt
import numpy as np
import cv2"""

os.chdir('/home/S319080106/anaconda3/envs/tensorflow-yyl/example/msa-gan')
from utils import *
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import cv2


model_ckpt = '/backup/this_disk_own_lhy/profile/geneGAN/trains2/plane2/sub_input/train_log_NoAeloss/model/model_084000.ckpt'
generateImg = GenerateImg(Model, model_ckpt)


def initial():
    GB_TYPES = ['bigtrack', 'smalltrack', 'cloud', 'skys']
    # GB_TYPES = ['cloud', 'skys']

    out_dir = '/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/test_pics/exp_default1/picked/generated'
    input_dir = '/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/test_pics/exp_default1/picked/orgs'
    #out_dir = 'mysdir/test_pics/exp_default1/picked/generated'
    #input_dir = 'mysdir/test_pics/exp_default1/picked/orgs'

    Bes = [[os.path.join(input_dir, dirname, filename) for filename in os.listdir(os.path.join(input_dir, dirname))]
           for dirname in GB_TYPES]

    Axs = [os.path.join(input_dir, 'plane', filename) for filename in os.listdir(os.path.join(input_dir, 'plane'))]
    img_Axs = generateImg.convert_paths2img(Axs)

    return GB_TYPES, out_dir, input_dir, Bes, Axs, img_Axs


def shiyan1():
    GB_TYPES, out_dir, input_dir, Bes, Axs, img_Axs = initial()
    for i in range(len(Bes)):
        Bes_ = Bes[i]
        img_Bes_ = generateImg.convert_paths2img(Bes_)
        for j in range(len(Axs)):
            img_Axs_ = img_Axs[j:j + 1]
            paths_Bx = [os.path.join(out_dir, GB_TYPES[i],
                                     Axs[j].split('/')[-1][:-4] + '_' + iBe.split('/')[-1][:-4] + '_Bx' + '.jpg')
                        for iBe in Bes_]
            paths_Ae = [os.path.join(out_dir, GB_TYPES[i],
                                     Axs[j].split('/')[-1][:-4] + '_' + iBe.split('/')[-1][:-4] + '_Ae' + '.jpg')
                        for iBe in Bes_]
            for k in range(len(Bes_)):
                imgBx, imgAe = generateImg.swap_attribute(img_Axs_, img_Bes_[k:k + 1])
                imsave_type32(imgBx, paths_Bx[k])
                imsave_type32(imgAe, paths_Ae[k])


def shiyan2():
    # out_canves = []
    GB_TYPES, out_dir, input_dir, Bes, Axs, img_Axs = initial()
    for i in range(len(Bes)):
        Bes_ = Bes[i]
        img_Bes_ = generateImg.convert_paths2img(Bes_)
        out_canve = np.zeros((0, 64 * len(Bes_) + 64, 3), dtype=np.float32)
        out_canve = np.concatenate([out_canve,
                                    np.concatenate([np.ones((64, 64, 3), dtype=np.float32) * 255]
                                                   + [img_Bes_[i] for i in range(len(img_Bes_))], axis=1)], axis=0)

        for j in range(len(Axs)):
            out_row = np.zeros((64, 0, 3), np.float32)
            img_Axs_ = img_Axs[j:j + 1]
            # out_paths = [os.path.join(out_dir, GB_TYPES[i], Axs[j].split('/')[-1][:-4]+'_'+iBe.split('/')[-1][:-4]+'.jpg')
            #          for iBe in Bes_]
            out_row = np.concatenate([out_row, img_Axs_[0]], axis=1)
            for k in range(len(Bes_)):
                out_img1, out_img2 = generateImg.swap_attribute(img_Axs_, img_Bes_[k:k + 1])
                # imsave_type32(out_img, out_paths[k])
                out_row = np.concatenate([out_row, out_img1], axis=1)
            out_canve = np.concatenate([out_canve, out_row], axis=0)
        # out_canves += [out_canve]
        out_path = os.path.join(out_dir, GB_TYPES[i] + '.jpg')
        imsave_type32(out_canve, out_path)

"""
def shiyan3(dir_out = 'mysdir/capsFromVidio/6thDepartment/generated/3',
            dir_AX = '/home/S317080041/mypro/pycharmFile/GAN_routine/Chinese_character/my_study/data/sections/test_pic/3',
            dir_BE = 'mysdir/capsFromVidio/6thDepartment/bkgd'):
    # dir_out = 'mysdir/capsFromVidio/6thDepartment/generated/3'
    # dir_AX = '/home/S317080041/mypro/pycharmFile/GAN_routine/Chinese_character/my_study/data/sections/test_pic/3'
    # dir_BE = 'mysdir/capsFromVidio/6thDepartment/bkgd'"""


def shiyan3(dir_out = '/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/capsFromVidio/6thDepartment/generated/3',
            dir_AX = '/backup1/S317080041/mypro/pycharmFile/GAN_routine/Chinese_character/my_study/data/sections/test_pic/3',
            dir_BE = '/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/capsFromVidio/6thDepartment/bkgd'):
    # dir_out = 'mysdir/capsFromVidio/6thDepartment/generated/3'
    # dir_AX = '/home/S317080041/mypro/pycharmFile/GAN_routine/Chinese_character/my_study/data/sections/test_pic/3'
    # dir_BE = 'mysdir/capsFromVidio/6thDepartment/bkgd'

    paths_AX = [os.path.join(dir_AX, i) for i in os.listdir(dir_AX)]
    paths_BE = [os.path.join(dir_BE, i) for i in os.listdir(dir_BE)]
    imgs_AX = generateImg.convert_paths2img(paths_AX)
    np.random.shuffle(imgs_AX)
    imgs_BE = generateImg.convert_paths2img(paths_BE)

    batch_size = 10
    for i in range(len(imgs_AX)//batch_size):
        batch_imgAX = imgs_AX[i*batch_size:i*batch_size+batch_size]
        batch_size_i = 10
        np.random.shuffle(imgs_BE)
        batch_imgBE = imgs_BE[:200]
        # paths_AE = [os.path.join(dir_out, str(i) + '_' +str(ii)) for ii in range(len(batch_imgBE))]
        for j in range(len(batch_imgBE)//batch_size_i):
            paths_AE = [os.path.join(dir_out, 'AE'+str(i)+'_'+str(j)+str(ii)+'.jpg') for ii in range(len(batch_imgBE)//batch_size_i)]
            paths_BX = [os.path.join(dir_out, 'BX'+str(i)+'_'+str(j)+str(ii)+'.jpg') for ii in range(len(batch_imgBE)//batch_size_i)]
            batch_imgBE_ = batch_imgBE[j*batch_size_i:j*batch_size_i+batch_size_i]
            imgBx, imgAe = generateImg.swap_attribute(batch_imgAX, batch_imgBE_, lens=batch_size_i)
            for k in range(batch_size_i):
                imsave_type32(imgBx[k], paths_BX[k])
                # imsave_type32(imgAe[k], paths_AE[k])
        print(i)

#
# shiyan3(dir_out = 'mysdir/capsFromVidio/6thDepartment/generated/3',
#             dir_AX = '/home/S317080041/mypro/pycharmFile/GAN_routine/Chinese_character/my_study/data/sections/test_pic/3',
#             dir_BE = 'mysdir/capsFromVidio/6thDepartment/bkgd')
#
# shiyan3(dir_out = 'mysdir/capsFromVidio/9thDepartment/generated/3',
#             dir_AX = '/home/S317080041/mypro/pycharmFile/GAN_routine/Chinese_character/my_study/data/sections/test_pic/3',
#             dir_BE = 'mysdir/capsFromVidio/9thDepartment/bkgd')
#
# shiyan3(dir_out = 'mysdir/capsFromVidio/house/generated/3',
#             dir_AX = '/home/S317080041/mypro/pycharmFile/GAN_routine/Chinese_character/my_study/data/sections/test_pic/3',
#             dir_BE = 'mysdir/capsFromVidio/house/bkgd')
#
# shiyan3(dir_out = 'mysdir/capsFromVidio/playground/generated/3',
#             dir_AX = '/home/S317080041/mypro/pycharmFile/GAN_routine/Chinese_character/my_study/data/sections/test_pic/3',
#             dir_BE = 'mysdir/capsFromVidio/playground/bkgd')
#
# shiyan3(dir_out = 'mysdir/capsFromVidio/shuishen/generated/3',
#             dir_AX = '/home/S317080041/mypro/pycharmFile/GAN_routine/Chinese_character/my_study/data/sections/test_pic/3',
#             dir_BE = 'mysdir/capsFromVidio/shuishen/bkgd')
#
# shiyan3(dir_out = 'mysdir/capsFromVidio/shuishen-close/generated/3',
#             dir_AX = '/home/S317080041/mypro/pycharmFile/GAN_routine/Chinese_character/my_study/data/sections/test_pic/3',
#             dir_BE = 'mysdir/capsFromVidio/shuishen-close/bkgd')

dict_src = {0: '3',
            1: '5',
            2: '6',
            3: '9',
            4: '12',
            5: '14',
            6: '15',
            7: '18'}
dict_dst = {0: '6thDepartment',
            1: '9thDepartment',
            2: 'house',
            3: 'playground',
            4: 'shuishen',
            5: 'shuishen-close',
            }

for i in range(len(dict_dst.keys())):
    for j in range(1, len(dict_src.keys())):
        dir_out = '/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/capsFromVidio/{}/generated/{}'.format(dict_dst[i], dict_src[j])
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        print(dir_out)
        dir_AX = '/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master//home/S317080041/mypro/pycharmFile/GAN_routine/Chinese_character/my_study/data/sections/test_pic/{}'.format(dict_src[j])
        dir_BE = '/backup1/S317080041/mypro/pycharmFile/GAN_routine/geneGAN/GeneGAN-master/mysdir/capsFromVidio/{}/bkgd'.format(dict_dst[i])
        shiyan3(dir_out=dir_out,
                dir_AX=dir_AX,
                dir_BE=dir_BE)




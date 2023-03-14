import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from models import get_generator, get_discriminator
import argparse
from glob import glob
import warnings

warnings.filterwarnings(action='ignore')

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=False, default=100, help='epochs')
parser.add_argument('--batchs', required=False, default=4, help='batchs')
parser.add_argument('--lr_g', required=False, default=0.0002, help='learning rate of generator')
parser.add_argument('--lr_d', required=False, default=0.0001, help='learning rate of discriminator')
parser.add_argument('--train_dir', required=False, default="./train/", help='directory of image to train / 학습 할 이미지 위치')
parser.add_argument('--load_model', required=False, default=True, help='load saved model / 저장된 모델 불러오기 (1: True, 0: False)')
parser.add_argument('--use_cpu', required=False, default=False, help='forced to use CPU only / CPU 만 이용해 학습하기 (1: True, 0: False)')
args = parser.parse_args()

epochs = args.epochs
batchs = args.batchs
lr_g = args.lr_g
lr_d = args.lr_d
train_dir =  args.train_dir
load_model =  args.load_model
use_cpu =  args.use_cpu

if use_cpu: os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 모델 불러오기 or 새로 생성하기
if load_model:

    if os.path.isfile('Generator.h5'):
        Generator= tf.keras.models.load_model('Generator.h5')
        print('Generator loaded')
    else:
        print('Cant load Generator')
        Generator = get_generator()

    if os.path.isfile('Discriminator.h5'):
        Discriminator = tf.keras.models.load_model('Discriminator.h5')
        print('Discriminator loaded')
    else:
        print('Cant load Discriminator')
        Discriminator = get_discriminator()

else:
    Generator = get_generator()
    Discriminator = get_discriminator()


def BGR2RGB(image):
    channels = tf.unstack(image, axis=-1)
    image    = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
    return image

imgs = []
mae = tf.keras.losses.MeanAbsoluteError()
bce = tf.losses.BinaryCrossentropy()
optim_g = tf.optimizers.Adam(lr_g, beta_1=0.9)
optim_d = tf.optimizers.Adam(lr_d, beta_1=0.9)
iter_count = 1
im_inx = glob(train_dir + "*.png") + glob(train_dir + "*.jpg")

for epoch in range(1, epochs+1):
    np.random.shuffle(im_inx)
    train_history = {'ssmi': list(), 'd_loss': list(), 'g_loss': list(), 'mse_loss': list(), 'vgg_loss': list(), 'adv_loss': list()}
    ssmi_scores = []
    for i in range(1, len(im_inx)+1):
        img = cv2.cvtColor(cv2.imread(im_inx[i-1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = tf.image.random_crop(img, (128,128,3)).numpy()
        imgs.append(img)

        if len(imgs) >= batchs or epoch == epochs:
            imgs_tensor_hr = np.array(imgs, dtype=np.float32)
            imgs_tensor_lr = tf.image.resize(imgs_tensor_hr, (32, 32), method=tf.image.ResizeMethod.BICUBIC).numpy()
            imgs_tensor_lr[imgs_tensor_lr >= 255] = 255
            imgs_tensor_lr[imgs_tensor_lr <= 0] = 0
            imgs_tensor_hr = imgs_tensor_hr / 127.5 -1
            imgs_tensor_lr = imgs_tensor_lr / 255
            imgs = []

            # Upate D
            imgs_tensor_sr = Generator(imgs_tensor_lr)
            with tf.GradientTape() as tape:
                hr_disc = Discriminator(imgs_tensor_hr)
                sr_disc = Discriminator(imgs_tensor_sr)
                D_RF = tf.sigmoid(hr_disc - tf.reduce_mean(sr_disc))
                D_FR = tf.sigmoid(sr_disc - tf.reduce_mean(hr_disc))
                loss_d = bce(tf.ones_like(input=D_RF), D_RF) + bce(tf.zeros_like(input=D_FR), D_FR)
                train_history['loss_d'] = float(loss_d)
            optim_d.minimize(loss_d, Discriminator.trainable_variables, tape = tape)

            #Update G
            with tf.GradientTape() as tape:
                imgs_tensor_sr = Generator(imgs_tensor_lr)
                hr_disc = Discriminator(imgs_tensor_hr)
                sr_disc = Discriminator(imgs_tensor_sr)
                loss_g = mae(imgs_tensor_hr, imgs_tensor_sr)
                train_history['loss_g'] = float(loss_g)

            optim_g.minimize(loss_g, Generator.trainable_variables, tape=tape)

            imgs_tensor_sr = imgs_tensor_sr.numpy()
            imgs_tensor_sr = (imgs_tensor_sr + 1) / 2
            imgs_tensor_sr[imgs_tensor_sr > 1] = 1
            imgs_tensor_sr[imgs_tensor_sr < 0] = 0
            imgs_tensor_hr = (imgs_tensor_hr + 1) / 2

            print("\repochs:", epoch, ", step:", i, len(im_inx), ", G loss:", round(float(loss_g), 5), ", D loss:", round(float(loss_d), 5), "ssim:", round(np.mean(tf.image.ssim(imgs_tensor_sr, imgs_tensor_hr, max_val = 1).numpy()), 5), "        ", end="")
            train_history['ssmi'].append(np.mean(tf.image.ssim(imgs_tensor_sr, imgs_tensor_hr, max_val = 1).numpy()))
            iter_count += 1
            
    print("\nepochs:", epoch, 
          'ssmi mean:', round(np.mean(train_history['ssmi']), 5), 
          'loss_d', round(np.mean(train_history['loss_d']), 5), 
          'loss_g', round(np.mean(train_history['loss_g']), 5), 
          'adv_loss', round(np.mean(train_history['adv_loss']), 5), 
          'vgg_loss', round(np.mean(train_history['vgg_loss']), 5), 
          'l1_loss', round(np.mean(train_history['l1_loss']), 5))
    
    Generator.save('Generator.h5')
    Discriminator.save('Discriminator.h5')
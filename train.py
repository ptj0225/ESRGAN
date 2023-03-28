import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from models import get_generator, get_discriminator, get_feature_extractor
import argparse
from glob import glob
import warnings
import util
import albumentations as A

warnings.filterwarnings(action='ignore')

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=False, default=1000, help='epochs')
parser.add_argument('--batchs', required=False, default=16, help='batchs')
parser.add_argument('--lr_g', required=False, default=0.0001, help='learning rate of generator')
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

# feature map 생성을 위한 feature extractor 선언
feature_extractor = get_feature_extractor()

feature_extractor.trainable = False
Discriminator.trainable = True
Generator.trainable = True

def BGR2RGB(image):
    channels = tf.unstack(image, axis=-1)
    image    = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
    return image

imgs = []
mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()
bce = tf.losses.BinaryCrossentropy()
optim_g = tf.keras.optimizers.Adam(lr_g, beta_1=0.9, beta_2=0.999)
optim_d = tf.keras.optimizers.Adam(lr_d, beta_1=0.9, beta_2=0.999)
iter_count = 1
im_inx = glob(train_dir + "*.png") + glob(train_dir + "*.jpg")
cv2.startWindowThread()
cv2.namedWindow('sample')

transform_hr = A.Compose([A.OneOf([A.RandomCrop(128,128),
                                   A.RandomCrop(256,256),
                                   A.RandomCrop(384,384),
                                   A.RandomCrop(512,512),
                                   A.RandomCrop(640,640)], p=1),
                        A.Resize(128,128, interpolation=cv2.INTER_CUBIC),
                        A.RandomRotate90(p=0.5),
                        A.HorizontalFlip(p=0.5)],)

transform_lr = A.Compose([
                        A.OneOf([A.GaussianBlur(always_apply=True),
                                   A.RingingOvershoot(always_apply=True)], p=0.7),
                        A.OneOf([A.Resize(64, 64, always_apply=True, interpolation=cv2.INTER_AREA),
                                 A.Resize(64, 64, always_apply=True, interpolation=cv2.INTER_CUBIC),
                                 A.Resize(64, 64, always_apply=True, interpolation=cv2.INTER_LINEAR)], p=1),
                        A.OneOf([A.GaussNoise(always_apply=True),
                                 A.ISONoise(always_apply=True),], p=0.7),
                        A.ImageCompression(quality_lower=70, p=0.7),
                        A.OneOf([A.GaussianBlur(always_apply=True),
                                   A.RingingOvershoot(always_apply=True)], p=0.7),
                        A.OneOf([A.Resize(32, 32, always_apply=True, interpolation=cv2.INTER_AREA),
                                 A.Resize(32, 32, always_apply=True, interpolation=cv2.INTER_CUBIC),
                                 A.Resize(32, 32, always_apply=True, interpolation=cv2.INTER_LINEAR)], p=1),
                        A.OneOf([A.GaussNoise(always_apply=True),
                                 A.ISONoise(always_apply=True)], p=0.7),
                        A.ImageCompression(quality_lower=70, p=0.7)
                                 ])


for epoch in range(1, epochs+1):
    np.random.shuffle(im_inx)
    train_history = {'ssmi': list(), 'psnr': list(), 'loss_d': list(), 'loss_g': list(), 'l1_loss': list(), 'vgg_loss': list(), 'adv_loss': list()}
    
    for i in range(1, len(im_inx)+1):
        try:
            img = cv2.cvtColor(cv2.imread(im_inx[i-1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            imgs.append(transform_hr(image=img)['image'])
        except:
            pass

        if len(imgs) >= batchs or epoch == epochs:
            imgs_tensor_hr = np.array(imgs, dtype=np.uint8)
            imgs_tensor_lr = np.array(list(map(lambda x: transform_lr(image=x)['image'], imgs_tensor_hr)), dtype=np.uint8)

            imgs_tensor_lr[imgs_tensor_lr >= 255] = 255
            imgs_tensor_lr[imgs_tensor_lr <= 0] = 0
            imgs_tensor_hr = imgs_tensor_hr / 127.5 -1
            imgs_tensor_lr = imgs_tensor_lr / 255
            imgs = []

            imgs_tensor_hr = tf.cast(imgs_tensor_hr, dtype=tf.float32)
            imgs_tensor_lr = tf.cast(imgs_tensor_lr, dtype=tf.float32)

            # Upate D
            imgs_tensor_sr = Generator(imgs_tensor_lr)
            with tf.GradientTape() as tape:

                # unfreeze bn
                Discriminator.trainable = True
                for layer in Discriminator.layers:
                    if isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.trainable = True

                hr_disc = Discriminator(imgs_tensor_hr)
                sr_disc = Discriminator(imgs_tensor_sr)
                D_RF = tf.sigmoid(hr_disc - tf.reduce_mean(sr_disc))
                D_FR = tf.sigmoid(sr_disc - tf.reduce_mean(hr_disc))
                loss_d = (bce(tf.ones_like(input=D_RF) - 0.00001, D_RF) + bce(tf.zeros_like(input=D_FR) + 0.00001, D_FR))/2

            optim_d.minimize(loss_d, Discriminator.trainable_variables, tape = tape)

            #Update G
            with tf.GradientTape() as tape:
                
                # freeze bn
                Discriminator.trainable = False
                for layer in Discriminator.layers:
                    if isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.trainable = False

                imgs_tensor_sr = Generator(imgs_tensor_lr)
                hr_disc = Discriminator(imgs_tensor_hr, training=False)
                sr_disc = Discriminator(imgs_tensor_sr, training=False)
                D_RF = tf.sigmoid(hr_disc - tf.reduce_mean(sr_disc))
                D_FR = tf.sigmoid(sr_disc - tf.reduce_mean(hr_disc))
                adv_loss = (bce(tf.zeros_like(input=D_RF), D_RF) + bce(tf.ones_like(input=D_FR), D_FR)) /2
                
                imgs_tensor_sr_feature_map = feature_extractor(127.5 * imgs_tensor_sr + 127.5) / 12.75
                imgs_tensor_hr_feature_map = feature_extractor(127.5 * imgs_tensor_hr + 127.5) / 12.75
                vgg_loss = mse(imgs_tensor_sr_feature_map, imgs_tensor_hr_feature_map)

                l1_loss = mae(imgs_tensor_sr, imgs_tensor_hr)

                loss_g = adv_loss * 0.005 + vgg_loss + l1_loss * 0.01

            optim_g.minimize(loss_g, Generator.trainable_variables, tape=tape)

            imgs_tensor_sr = imgs_tensor_sr.numpy()
            imgs_tensor_sr = (imgs_tensor_sr + 1) / 2
            imgs_tensor_sr[imgs_tensor_sr > 1] = 1
            imgs_tensor_sr[imgs_tensor_sr < 0] = 0
            imgs_tensor_hr = (imgs_tensor_hr + 1) / 2

            #  학습중인 이미지 보여주기
            sample_img = np.concatenate([cv2.resize(np.array(imgs_tensor_lr[0]), dsize=(128, 128), interpolation=cv2.INTER_CUBIC), 
                                        imgs_tensor_sr[0], 
                                        imgs_tensor_hr[0]],
                                        axis=1)
            sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
            sample_img = cv2.resize(sample_img, dsize=(900,300), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(winname = 'sample', mat=sample_img)
            cv2.waitKey(1)

            train_history['loss_d'].append(round(float(loss_d), 5))
            train_history['adv_loss'].append(round(float(adv_loss), 5))
            train_history['vgg_loss'].append(round(float(vgg_loss), 5))
            train_history['l1_loss'].append(round(float(l1_loss), 5))
            train_history['loss_g'].append(round(float(loss_g), 5))
            train_history['ssmi'].append(round(float(np.mean(tf.image.ssim(imgs_tensor_sr, imgs_tensor_hr, max_val = 1).numpy())), 5))
            train_history['psnr'].append(round(float(np.mean(tf.image.psnr(imgs_tensor_sr, imgs_tensor_hr, max_val = 1).numpy())), 5))

            print("\repoch, iteration:", epoch, iter_count, ", step:", i, len(im_inx), ", loss_g:", train_history['loss_g'][-1], ",loss_d", train_history['loss_d'][-1], "ssim:", train_history['ssmi'][-1], ", psnr:", train_history['psnr'][-1], end="")
            train_history['ssmi'].append(np.mean(tf.image.ssim(imgs_tensor_sr, imgs_tensor_hr, max_val = 1).numpy()))
            
            # lr halved
            if iter_count in [50000, 100000, 200000, 300000]:
                optim_d.lr = optim_d.lr / 2
                optim_g.lr = optim_g.lr / 2
                print(f"lr chaged ", optim_d.lr.numpy(), optim_g.lr.numpy())

            iter_count += 1

    Generator.save('Generator.h5')
    Discriminator.save('Discriminator.h5')

    print("\nepochs:", epoch, 
          'iteration', iter_count,
          'ssmi mean:', round(np.mean(train_history['ssmi']), 5), 
          'psnr mean:', round(np.mean(train_history['psnr']), 5), 
          'loss_d', round(np.mean(train_history['loss_d']), 5), 
          'loss_g', round(np.mean(train_history['loss_g']), 5), 
          'adv_loss', round(np.mean(train_history['adv_loss']), 5), 
          'vgg_loss', round(np.mean(train_history['vgg_loss']), 5), 
          'l1_loss', round(np.mean(train_history['l1_loss']), 5))
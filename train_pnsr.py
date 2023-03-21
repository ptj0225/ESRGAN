import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from models import get_generator, get_discriminator, get_feature_extractor
import argparse
from glob import glob
import warnings
import albumentations as A

warnings.filterwarnings(action='ignore')

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=False, default=100, help='epochs')
parser.add_argument('--batchs', required=False, default=32, help='batchs')
parser.add_argument('--lr_g', required=False, default=0.0001, help='learning rate of generator')
parser.add_argument('--train_dir', required=False, default="./train/", help='directory of image to train / 학습 할 이미지 위치')
parser.add_argument('--load_model', required=False, default=True, help='load saved model / 저장된 모델 불러오기 (1: True, 0: False)')
parser.add_argument('--use_cpu', required=False, default=False, help='forced to use CPU only / CPU 만 이용해 학습하기 (1: True, 0: False)')
args = parser.parse_args()

epochs = args.epochs
batchs = args.batchs
lr_g = args.lr_g
train_dir =  args.train_dir
load_model =  args.load_model
use_cpu =  args.use_cpu

if use_cpu: os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 모델 불러오기 or 새로 생성하기
if load_model:

    if os.path.isfile('Generator_PSNR.h5'):
        Generator= tf.keras.models.load_model('Generator_PSNR.h5')
        print('Generator loaded')
    else:
        print('Cant load Generator')
        Generator = get_generator()
else:
    Generator = get_generator()
Generator.trainable = True


imgs = []
mae = tf.keras.losses.MeanAbsoluteError()
optim_g = tf.keras.optimizers.experimental.AdamW(lr_g, beta_1=0.9, beta_2=0.999)
iter_count = 1
im_inx = glob(train_dir + "*.png") + glob(train_dir + "*.jpg")
best_score = 0
cv2.startWindowThread()
cv2.namedWindow('sample')

transform_hr = A.Compose([A.RandomRotate90(p=0.5),
                          A.HorizontalFlip(p=0.5)],)
transform_lr = A.Compose([A.Resize(32,32, p=1, always_apply=True, interpolation=cv2.INTER_CUBIC)])


for epoch in range(1, epochs+1):
    np.random.shuffle(im_inx)
    train_history = {'ssmi': list(), 'psnr': list(), 'loss_g': list(), 'l1_loss': list()}

    for i in range(1, len(im_inx)+1):
        img = cv2.cvtColor(cv2.imread(im_inx[i-1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        imgs.append(img)

        if len(imgs) >= batchs or epoch == epochs:
            imgs_tensor_hr = np.array(list(map(lambda x: transform_hr(image=x)['image'], imgs)), dtype=np.uint8)
            imgs_tensor_lr = np.array(list(map(lambda x: transform_lr(image=x)['image'], imgs_tensor_hr)), dtype=np.uint8)

            imgs_tensor_lr[imgs_tensor_lr >= 255] = 255
            imgs_tensor_lr[imgs_tensor_lr <= 0] = 0
            imgs_tensor_hr = imgs_tensor_hr / 127.5 -1
            imgs_tensor_lr = imgs_tensor_lr / 255
            imgs = []

            imgs_tensor_hr = tf.cast(imgs_tensor_hr, dtype=tf.float32)
            imgs_tensor_lr = tf.cast(imgs_tensor_lr, dtype=tf.float32)

            #Update G
            with tf.GradientTape() as tape:
                imgs_tensor_sr = Generator(imgs_tensor_lr)
                l1_loss = mae(imgs_tensor_sr, imgs_tensor_hr)
                loss_g = l1_loss * 10

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
            sample_img = cv2.resize(sample_img, dsize=(900,300), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(winname = 'sample', mat=sample_img)
            cv2.waitKey(1)

            if iter_count % 100 == 0:
                Generator.save('Generator_PSNR.h5')

            train_history['l1_loss'].append(round(float(l1_loss), 5))
            train_history['loss_g'].append(round(float(loss_g), 5))
            train_history['ssmi'].append(round(float(np.mean(tf.image.ssim(imgs_tensor_sr, imgs_tensor_hr, max_val = 1).numpy())), 5))
            train_history['psnr'].append(round(float(np.mean(tf.image.psnr(imgs_tensor_sr, imgs_tensor_hr, max_val = 1).numpy())), 5))

            print("\repochs:", epoch, ", step:", i, len(im_inx), ", loss_g:", train_history['loss_g'][-1], "ssim:", train_history['ssmi'][-1], ", psnr:", train_history['psnr'][-1], end="")
            train_history['ssmi'].append(np.mean(tf.image.ssim(imgs_tensor_sr, imgs_tensor_hr, max_val = 1).numpy()))
            iter_count += 1

            # lr halved
            if iter_count % 200000 == 0:
                optim_g.lr = optim_g.lr / 2
                print(f"lr chaged ", optim_g.lr.numpy())

    print("\repochs:", epoch, 
          'ssmi mean:', round(np.mean(train_history['ssmi']), 5), 
          'psnr mean:', round(np.mean(train_history['psnr']), 5), 
          'loss_g', round(np.mean(train_history['loss_g']), 5), 
          'l1_loss', round(np.mean(train_history['l1_loss']), 5))
    
    if best_score <= np.mean(train_history['psnr']):
        best_score = np.mean(train_history['psnr'])
        Generator.save('Generator_PSNR.h5')
        print('Generator_PSNR.h5 saved')
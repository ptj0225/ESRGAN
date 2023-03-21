import tensorflow as tf
from tensorflow.keras import layers

def ResDenseBlock(x):
    x1 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="SAME")(x)
    x1 = layers.LeakyReLU(alpha=0.2)(x1)
    x2 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="SAME")(layers.concatenate([x, x1], 3))
    x2 = layers.LeakyReLU(alpha=0.2)(x2)
    x3 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="SAME")(layers.concatenate([x, x1, x2], 3))
    x3 = layers.LeakyReLU(alpha=0.2)(x3)
    x4 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="SAME")(layers.concatenate([x, x1, x2, x3], 3))
    x4 = layers.LeakyReLU(alpha=0.2)(x4)
    x5 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="SAME")(layers.concatenate([x, x1, x2, x3, x4], 3))

    return x5

def RRDB(input):
    x = input
    x = ResDenseBlock(x)
    x = ResDenseBlock(x)
    x = ResDenseBlock(x)
    output = input + x * 0.2
    return output


def get_generator(eparable_cnn=False):

    CNN = layers.Conv2D

    input = tf.keras.Input(shape=(None,None,3))
    output = layers.Conv2D(64, (9, 9), strides=(1, 1), padding='same')(input)
    output = layers.LeakyReLU(alpha=0.2)(output)
    output_ = output

    for _ in range(16): output = RRDB(output)

    output = CNN(64, 3, padding='same')(output)

    output = layers.Add()([output, output_])

    h = tf.shape(output)[1]
    w = tf.shape(output)[2]
    output = tf.image.resize(output, [h * 2, w * 2], method='nearest')
    output = layers.Conv2D(64, 3, padding='same')(output)
    output = layers.LeakyReLU(alpha=0.2)(output)

    output = tf.image.resize(output, [h * 4, w * 4], method='nearest')
    output = layers.Conv2D(64, 3, padding='same')(output)
    output = layers.LeakyReLU(alpha=0.2)(output)
    
    output = layers.Conv2D(64, 3, padding='same')(output)
    output = layers.LeakyReLU(alpha=0.2)(output)

    output = layers.Conv2D(3, 9, padding='same')(output)
    
    return tf.keras.models.Model(input, output)


def dis_block(x,k,n,s,include_bn=True):
  x= layers.Conv2D(n, k, strides=s, padding='same')(x)

  if include_bn: x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  
  return x


def get_discriminator(include_bn=True):
    input = tf.keras.Input(shape=(128,128,3))
    output = layers.Conv2D(64, 3, 1, padding='same')(input)
    output = layers.LeakyReLU(alpha=0.2)(output)

    output = dis_block(output,3,64,2,include_bn)
    output = dis_block(output,3,128,1,include_bn)
    output = dis_block(output,3,128,2,include_bn)
    output = dis_block(output,3,256,1,include_bn)
    output = dis_block(output,3,256,2,include_bn)
    output = dis_block(output,3,512,1,include_bn)
    output = dis_block(output,3,512,2,include_bn)
    output = dis_block(output,3,512,1,include_bn)
    output = dis_block(output,3,512,2,include_bn)

    output = layers.Flatten()(output)
    output = layers.Dense(512)(output)
    output = layers.LeakyReLU(alpha=0.2)(output)
    output = layers.Dense(1)(output)
    
    return tf.keras.Model(input, output)


def vgg19():
    input = tf.keras.Input(shape=(None,None,3))
    x = input

    for _ in range(2): x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="SAME", activation="ReLU")(x)
    x = layers.MaxPooling2D(2, strides=(1, 1), padding="SAME")(x)

    for _ in range(2): x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", activation="ReLU")(x)
    x = layers.MaxPooling2D(2, strides=(1, 1), padding="SAME")(x)

    for _ in range(4): x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="SAME", activation="ReLU")(x)
    x = layers.MaxPooling2D(2, strides=(1, 1), padding="SAME")(x)

    for _ in range(4): x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding="SAME", activation="ReLU")(x)
    x = layers.MaxPooling2D(2, strides=(1, 1), padding="SAME")(x)
    
    for _ in range(3): x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding="SAME", activation="ReLU")(x)
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding="SAME", activation=None)(x)
    x = layers.MaxPooling2D(2, strides=(1, 1), padding="SAME")(x)

    return tf.keras.Model(input, x)

def get_feature_extractor():
    input = tf.keras.Input(shape=(128,128,3))
    x = input
    x = tf.keras.applications.vgg19.preprocess_input(x)
    for _ in range(2): x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="SAME", activation="ReLU")(x)
    x = layers.MaxPooling2D((2, 2), padding="SAME")(x)

    for _ in range(2): x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", activation="ReLU")(x)
    x = layers.MaxPooling2D((2, 2), padding="SAME")(x)

    for _ in range(4): x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="SAME", activation="ReLU")(x)
    x = layers.MaxPooling2D((2, 2), padding="SAME")(x)

    for _ in range(4): x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding="SAME", activation="ReLU")(x)
    x = layers.MaxPooling2D((2, 2), padding="SAME")(x)
    
    for _ in range(3): x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding="SAME", activation="ReLU")(x)
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding="SAME", activation=None)(x)

    model = tf.keras.Model(input, x)
    model.set_weights(tf.keras.applications.VGG19(include_top=False, weights="imagenet").weights)
    model.trainable = False

    return model
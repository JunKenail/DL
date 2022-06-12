"""
作者：刘俊
日期：2022年5月10日
github地址：
"""

""" denseblock.py（58行）
此模块有设计封装好的denseblock_3D和transitionblock_3D等网络块
"""

from keras.layers import *
from keras.regularizers import l2

# 或from tensorflow.keras.... import ...
# 或from tensorflow.python.keras.... import ...


# 层的命名不能重复


def convblock_3d(input_x, filters, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(epsilon=1.1e-5)(input_x)
    # x = ReLU()(x)
    x = LeakyReLU()(x)

    if bottleneck:
        x = Conv3D(filters=filters * 4, kernel_size=1, kernel_initializer='he_normal',  # bottleneck
               padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)

        x = BatchNormalization(epsilon=1.1e-5)(x)
        # x = ReLU()(x)
        x = LeakyReLU()(x)
    x = Conv3D(filters=filters, kernel_size=3, kernel_initializer='he_normal',
               padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def denseblock_3d(x, nb_layers, filters, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    for i in range(nb_layers):
        cb = convblock_3d(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        if grow_nb_filters:
            filters += growth_rate
    return x, filters


def transitionlayers_3d(input_x, filters, compression_rate=1.0, weight_decay=1e-4):
    x = BatchNormalization(epsilon=1.1e-5)(input_x)
    # x = ReLU()(x)
    x = LeakyReLU()(x)
    x = Conv3D(filters=filters * compression_rate, kernel_size=1, kernel_initializer='he_normal',
               padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling3D(pool_size=2, strides=2)(x)
    return x

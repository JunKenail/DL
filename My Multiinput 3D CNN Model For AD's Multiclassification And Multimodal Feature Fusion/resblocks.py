"""
作者：刘俊
日期：2022年5月10日
github地址：
"""

""" resblock.py（177行）
此模块为设计封装好的各种残差结构块（resblock）
"""

from keras.layers import *


# 或from tensorflow.keras.... import ...
# 或from tensorflow.python.keras.... import ...

# 层的命名不能重复

def BN_Relu_Conv2Dor3D(filters, kernel_size, strides, input_layer, conv_dimension=3, dropout_rate=None):
    """ pre-activation: BN+ReLU before Conv
    """
    x = BatchNormalization()(input_layer)
    # x = ReLU()(x)
    x = LeakyReLU()(x)
    if conv_dimension == 2:
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    else:
        x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    return x


def BasicResblock(input_x, filters, downsampling=False, conv_dimension=3, dropout_rate=None):
    strides = 2 if downsampling else 1
    x = BN_Relu_Conv2Dor3D(filters, 3, strides, input_x, conv_dimension)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = BN_Relu_Conv2Dor3D(filters, 3, 1, x, conv_dimension)
    y = Add()([x, input_x])
    return y


def BasicInceptionResblock(input_x, filters, downsampling=False, conv_dimension=3, dropout_rate=None):
    strides = 2 if downsampling else 1
    x = BN_Relu_Conv2Dor3D(filters, 3, strides, input_x, conv_dimension)
    input_x = BN_Relu_Conv2Dor3D(filters, 3, strides, input_x, conv_dimension)  # 支路
    x = BN_Relu_Conv2Dor3D(filters, 3, 1, x, conv_dimension)
    y = Add()([x, input_x])
    if dropout_rate:
        y = Dropout(dropout_rate)(y)
    return y


def BottleneckResblock(input_x, filters, downsampling=False, k=4, conv_dimension=3, dropout_rate=None):
    strides = 2 if downsampling else 1
    x = BN_Relu_Conv2Dor3D(filters, 1, strides, input_x, conv_dimension)
    x = BN_Relu_Conv2Dor3D(filters, 3, 1, x, conv_dimension)
    x = BN_Relu_Conv2Dor3D(filters * k, 1, 1, x, conv_dimension)
    y = Add()([x, input_x])
    if dropout_rate:
        y = Dropout(dropout_rate)(y)
    return y


def BottleneckInceptionResblock(input_x, filters, downsampling=False, k=4, conv_dimension=3, dropout_rate=None):
    strides = 2 if downsampling else 1
    x = BN_Relu_Conv2Dor3D(filters, 1, strides, input_x, conv_dimension)
    x = BN_Relu_Conv2Dor3D(filters, 3, 1, x, conv_dimension)
    x = BN_Relu_Conv2Dor3D(filters * k, 1, 1, x, conv_dimension)
    input_x = BN_Relu_Conv2Dor3D(filters * k, 1, strides, input_x, conv_dimension)  # 支路
    y = Add()([x, input_x])
    if dropout_rate:
        y = Dropout(dropout_rate)(y)
    return y


# 下面的模块是最初的设计，重新设计编写之后下面的模块便没有使用。
def Conv2Dor3D_BN_Relu(filters, kernel_size, strides, input_layer, name, conv_dimension=3, dropout_rate=None):
    """ Conv2D/3D-BatchNormalization-Relu构造的Conv块
    Args:
        filters: 卷积核的数目
        kernel_size: 卷积核的大小，可以是单个整数，或2个（2D）/3个（3D整数构成的list/tuple
        strides: 卷积的步长，可以是单个整数，或2个（2D）/3个（3D整数构成的list/tuple
        conv_dimension: 卷积的维度
        dropout_rate: dropout的比率
    """
    if conv_dimension == 2:
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                   name=name + '_ConvBlock_Conv2D')(input_layer)
    else:
        x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                   name=name + '_ConvBlock_Conv3D')(input_layer)
    x = BatchNormalization(name=name + '_ConvBlock_BN_Relu')(x)
    x = ReLU()(x)
    # x = LeakyReLU()(x)
    if dropout_rate:  # 所有Dropout都在最后进行
        x = Dropout(dropout_rate)(x)
    return x


def Conv2Dor3D_BN_Relu_with_one_shotcut(filters, kernel_size, strides, input_layer, name, conv_dimension=3,
                                         dropout_rate=None):
    # 一个Conv块+一路shotcut
    x = Conv2Dor3D_BN_Relu(filters, kernel_size, strides, input_layer, conv_dimension, name)
    x = Add()([x, input_layer])
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def Conv2Dor3D_BN_Relu_with_two_shotcuts(filters, kernel_size, strides, input_layer, name, conv_dimension=3,
                                          dropout_rate=None):
    # 两个Conv块+两路shotcut
    x = Conv2Dor3D_BN_Relu_with_one_shotcut(filters, kernel_size, strides, input_layer, conv_dimension, name)
    x = Conv2Dor3D_BN_Relu(filters, kernel_size, strides, x, conv_dimension, name)
    x = Add()([x, input_layer])
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def resblock_a_2dor3d(input_x, filters, name, conv_dimension=3, dropout_rate=None):
    x = Conv2Dor3D_BN_Relu(filters, 3, 1, input_x, name + '_' + str(1), conv_dimension)
    x = Conv2Dor3D_BN_Relu(filters, 3, 1, x, name + '_' + str(2), conv_dimension)
    y = Add()([x, input_x])
    if dropout_rate:
        y = Dropout(dropout_rate)(y)
    return y


def resblock_a_2_2dor3d(input_x, filters, name, conv_dimension=3, dropout_rate=None):
    x = Conv2Dor3D_BN_Relu(filters, 3, 2, input_x, name + '_' + str(1), conv_dimension)
    x = Conv2Dor3D_BN_Relu(filters, 3, 1, x, name + '_' + str(2), conv_dimension)
    y = Add()([x, input_x])
    if dropout_rate:
        y = Dropout(dropout_rate)(y)
    return y


def resblock_b_2dor3d(input_x, filters, name, conv_dimension=3, dropout_rate=None):
    x = Conv2Dor3D_BN_Relu(filters, 3, 2, input_x, name + '_' + str(1), conv_dimension)
    x = Conv2Dor3D_BN_Relu(filters, 3, 1, x, name + '_' + str(2), conv_dimension)
    input_x = Conv2Dor3D_BN_Relu(filters, 3, 2, input_x, name + '_shotcut', conv_dimension)  # 支路
    y = Add()([x, input_x])
    if dropout_rate:
        y = Dropout(dropout_rate)(y)
    return y


def resblock_c_2dor3d(input_x, filters, name, conv_dimension=3, dropout_rate=None):
    x = Conv2Dor3D_BN_Relu(filters, 1, 1, input_x, name + '_' + str(1), conv_dimension)
    x = Conv2Dor3D_BN_Relu(filters, 3, 1, x, name + '_' + str(2), conv_dimension)
    x = Conv2Dor3D_BN_Relu(filters * 4, 1, 1, x, name + '_' + str(3), conv_dimension)  # filters * 4 对应上个block改成256可否？
    y = Add()([x, input_x])
    if dropout_rate:
        y = Dropout(dropout_rate)(y)
    return y


def resblock_c_2_2dor3d(input_x, filters, name, conv_dimension=3, dropout_rate=None):
    x = Conv2Dor3D_BN_Relu(filters, 1, 2, input_x, name + '_' + str(1), conv_dimension)
    x = Conv2Dor3D_BN_Relu(filters, 3, 1, x, name + '_' + str(2), conv_dimension)
    x = Conv2Dor3D_BN_Relu(filters * 4, 1, 1, x, name + '_' + str(3), conv_dimension)
    y = Add()([x, input_x])
    if dropout_rate:
        y = Dropout(dropout_rate)(y)
    return y


def resblock_d_2dor3d(input_x, filters, name, conv_dimension=3, dropout_rate=None):
    x = Conv2Dor3D_BN_Relu(filters, 1, 2, input_x, name + '_' + str(1), conv_dimension)
    x = Conv2Dor3D_BN_Relu(filters, 3, 1, x, name + '_' + str(2), conv_dimension)
    x = Conv2Dor3D_BN_Relu(filters * 4, 1, 1, x, name + '_' + str(3), conv_dimension)
    input_x = Conv2Dor3D_BN_Relu(filters * 4, 1, 2, input_x, name + '_shotcut', conv_dimension)  # 支路
    y = Add()([x, input_x])
    if dropout_rate:
        y = Dropout(dropout_rate)(y)
    return y
    
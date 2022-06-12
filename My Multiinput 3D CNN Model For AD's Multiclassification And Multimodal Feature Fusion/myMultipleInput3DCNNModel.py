"""
作者：刘俊
日期：2022年5月10日
github地址：
"""

""" myMultipleInput3DCNNModel.py（236行）
此模型为我设计的多输入3DCNN模型，两个输入端、一个融合输出端、整个网络都进行了封装。
"""

import os
import resblocks
import denseblocks
# import tensorflow as tf
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import losses
from keras import callbacks
from keras.utils.vis_utils import plot_model, model_to_dot
import matplotlib.pyplot as plt


# 或from tensorflow.keras.... import ...
# 或from tensorflow.python.keras.... import ...




def denselayers_for_mri_before_fusion(mri_inputshape, nb_denseblock, filters, growth_rate, nb_layers_per_denseblock):
    """ MRI端
    总层数：99
    # 以denseblock和transitionlayers为核心
    """
    # MRI输入端
    mri_input = Input(mri_inputshape)  # mri_inputshape = (196, 256, 256, 1)
    mri_conv = Conv3D(64, 7, 2, padding='same')(mri_input)  # => (98, 128, 128, 1)
    # mri_conv = Conv3D(64, 3, 1, padding='same')(mri_input)
    mri_conv = BatchNormalization()(mri_conv)
    mri_conv = ReLU()(mri_conv)
    # mri_conv = LeakyReLU()(mri_conv)
    mri_maxpooling = MaxPool3D(pool_size=3, strides=2)(mri_conv)  # => (49, 64, 64, 1)

    # MRI中间层
    mri_x = mri_maxpooling
    bottleneck = True
    dropout_rate = None
    compression_rate = 1.0  # 压缩率
    for i in range(nb_denseblock-1):
        mri_x, filters = denseblocks.denseblock_3d(mri_x, nb_layers_per_denseblock[i], filters, growth_rate,
                                                   bottleneck=bottleneck, dropout_rate=dropout_rate)
        mri_x = denseblocks.transitionlayers_3d(mri_x, filters, compression_rate=compression_rate)
        filters = int(filters * compression_rate)
    mri_x, filters = denseblocks.denseblock_3d(mri_x, nb_layers_per_denseblock[-1], filters, growth_rate,
                                               bottleneck=bottleneck, dropout_rate=dropout_rate)
    mri_x = BatchNormalization()(mri_x)
    mri_x = ReLU()(mri_x)
    # mri_x = LeakyReLU()(mri_x)

    # MRI输出端
    mri_output = GlobalAveragePooling3D()(mri_x)  # 全局平均池化

    return mri_input, mri_output


def reslayers_for_pet_before_fusion(pet_inputshape):
    """ PET端
    # 以resblock为核心
    注：详细参数信息请见对应“SUMMARY_....txt”文件
    """
    # PET输入端
    pet_input = Input(pet_inputshape)  # pet_inputshape = (160, 160, 96, 1)
    pet_conv = Conv3D(64, 7, 2, padding='same')(pet_input)  # => (80, 80, 48, 1)
    # pet_conv = Conv3D(32, 3, 1, padding='same')(pet_input)
    pet_conv = BatchNormalization()(pet_conv)
    pet_conv = ReLU()(pet_conv)
    # pet_conv = LeakyReLU()(pet_conv)
    pet_maxpooling = MaxPool3D(pool_size=2, strides=2)(pet_conv)  # => (40, 40, 24, 1)

    # PET中间层
    pet_x = resblocks.BasicResblock(pet_maxpooling, 64)
    pet_x = resblocks.BasicResblock(pet_x, 64)
    pet_x = resblocks.BasicResblock(pet_x, 64)  # => (40, 40, 24, 1)
    pet_x = resblocks.BasicInceptionResblock(pet_x, 128, downsampling=True)
    pet_x = resblocks.BasicResblock(pet_x, 128)
    pet_x = resblocks.BasicResblock(pet_x, 128)  # => (20, 20, 12, 1)
    pet_x = resblocks.BottleneckInceptionResblock(pet_x, 128, downsampling=True, k=2)
    pet_x = resblocks.BottleneckResblock(pet_x, 128, k=2)
    pet_x = resblocks.BottleneckResblock(pet_x, 128, k=2)  # => (10, 10, 6, 1)
    pet_x = resblocks.BottleneckInceptionResblock(pet_x, 256, downsampling=True, k=2)
    pet_x = resblocks.BottleneckResblock(pet_x, 256, k=2)
    pet_x = resblocks.BottleneckResblock(pet_x, 256, k=2)  # => (5, 5, 3, 1)
    pet_x = BatchNormalization()(pet_x)
    pet_x = ReLU()(pet_x)
    # pet_x = LeakyReLU()(pet_x)
    # PET输出端
    pet_output = GlobalAveragePooling3D()(pet_x)  # 全局平均池化

    return pet_input, pet_output


def fusionlayers(mri_output, pet_output, nb_classes):
    """ 融合输出端
    注：详细参数信息请见对应“SUMMARY_....txt”文件
    """
    mripet_fusion = concatenate([mri_output, pet_output], axis=-1)
    
    bn1 = BatchNormalization()(mripet_fusion)
    fc1 = Dense(1024, activation=None)(bn1)
    fc1 = LeakyReLU()(fc1)
    # fc1 = Dropout(0.1)(fc1)
    bn2 = BatchNormalization()(fc1)
    fc2 = Dense(512, activation=None)(bn2)
    fc2 = LeakyReLU()(fc2)
    # fc2 = Dropout(0.1)(fc2)
    bn3 = BatchNormalization()(fc2)
    output = Dense(nb_classes, activation='softmax')(bn3)

    return output


class myMultipleInput3DCNNModel:
    def __init__(self):
        self.modelpath = ''
        self.modelweightspath = ''
        self.log_dir = ''  # TensorBoard可视化需要的文件目录。
        print('The model initialized successfully!')

    def built_model(self, mri_inputshape, pet_inputshape, nb_classes):
        """ 构建并返回MRI、PET的多输入3DCNN模型，利用该模型实现MRI、PET的多模态特征融合和分类
        Args:
            imgmri: 输入mri图像数据的shape (196,256,256,1)
            imgpet: 输入pet图像数据的shape (160,160,96,1)
            nb_classes: 分类数目
        return model
        注：详细参数信息请见对应“SUMMARY_....txt”文件
        """
        self.mri_inputshape = mri_inputshape
        self.pet_inputshape = pet_inputshape
        self.nb_classes = nb_classes


        mri_input, mri_output = denselayers_for_mri_before_fusion(mri_inputshape, 4, 64, 32, [4, 4, 4, 4])  # 4, 64, 32, [4, 6, 8, 6]
        pet_input, pet_output = reslayers_for_pet_before_fusion(pet_inputshape)
        output = fusionlayers(mri_output, pet_output, nb_classes)

        self.model = Model(inputs=[mri_input, pet_input], outputs=output, name='My_Multiple_Input_3DCNN_Model')
        print('The model built successfully!')
        self.lr = 0.001
        adm = Adam(lr=self.lr)  # 初始学习率
        self.lr_reduce = True
        self.lr_reduce_factor = 0.8
        self.model.compile(optimizer=adm, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        print('The model compiled successfully!')
        return self.model

    def trian_and_save_model(self, x_train, y_train):  # x_train = [np.array(mri_trainingdata), np.array(pet_trainingdata)]
        """ 关于模型的几个超参数：学习率、batchsize、validation_split比率、epoch
            值得训练比较和调试。
        """
        print('Training begins:')
        # self.modelpath = 'My_Multiple_Input_3DCNN_Model'  # 保存整个
        if not os.path.isdir(self.modelpath):
            os.mkdir(self.modelpath)
        # self.modelweightspath = 'My_Multiple_Input_3DCNN_Model_best_weights.h5'  # 保存模型的权重
        checkpoint_cb = callbacks.ModelCheckpoint(self.modelweightspath, monitor='val_loss', verbose=1, save_best_only=True,
                                                  mode='max')  # , save_freq=1 
        # EarlyStop = callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)  # 早停机制
        ReduceLROnP = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=self.lr_reduce_factor, patience=2, verbose=1, min_lr=10e-6)  # 动态调整学习率, factor为下降因子
        # self.log_dir ='logs'
        self.epochs = 50
        self.batch_size = 8
        print('Parameter setting: lr=', self.lr,', lr_reduce=', self.lr_reduce, ', ReduceLROnPlateau by factor of ', self.lr_reduce_factor, ', epochs=', self.epochs, ', batch_size=', self.batch_size)
        TensorB = callbacks.TensorBoard(log_dir=self.log_dir)  # 使用TensorBoard对结果进行可视化。
        self.history = self.model.fit(x_train, y_train, shuffle=True, validation_split=0.3, epochs=self.epochs, 
                      batch_size=self.batch_size, verbose=2, callbacks=[checkpoint_cb, ReduceLROnP, TensorB])  # , EarlyStop
        # e.g: validation_split = 0.1, 0.2, 0.3  epochs = 32, 16, 10, 8   batch_size = 2, 4, 8, 16
        """ # PS：
        # 验证集占比对训练效果的影响取决于训练集的数据量。
        # 若训练集较小并担心验证集占比过大，加大epochs即可有效地利用所有数据，因为每批次训练前都会打乱数据。
        # epochs的设置可以结合训练效果，太小可能训练效果不好，而太大可能后面的训练是多余的，并没有什么提升效果。
        # 但实际最终的模型训练效果还是取决于训练集的大小，训练集越大过拟合可能性就越小、泛化能力就越强、会更鲁棒和健壮。
        # 另外，batch_size的设置也会影响训练的效果，
        # 如果有硬件条件运行，可以尝试使用更大的batch_size，如8、16、32，同时用更大的数据集去训练，或许会得到更好的效果。
        # 因为每个batch_size更新一次网络参数，理论上batch_size越大，信息根据就就越多，网络参数的更新方向就越准确，但具体
        # 那个值更好，可以用实验去进行比较。
        """
        self.model.save(self.modelpath)  # 保存完整模型
        print('Training ends!')
        print('''The trained model's weights saved successfully in ''' + self.modelweightspath)
        print('''The trained model saved successfully in ''' + self.modelpath)

    def model_summary(self):
        self.model.summary()

    def trian_and_save_model_based_on_existing_model(self, existingmodelpath, x_train, y_train):  # 加载已有模型继续进行训练
        print('Continue training begins:')
        self.lr = 0.0005
        adm = Adam(lr=self.lr)
        self.lr_decay = 'False'
        self.model.compile(optimizer=adm, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        self.model = load_model(existingmodelpath)
        checkpoint_cb = callbacks.ModelCheckpoint(self.modelweightspath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                                  mode='max')  # , save_freq=1
        # EarlyStop = callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        ReduceLROnP = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=1, min_lr=10e-6)
        # self.log_dir ='logs'
        TensorB = callbacks.TensorBoard(log_dir=self.log_dir)  # 使用TensorBoard对结果进行可视化。
        self.epochs = 16
        self.batch_size = 8
        print('Parameter setting: lr=', self.lr,', lr_decay=', self.lr_decay, ', epochs=', self.epochs, ', batch_size=', self.batch_size)
        self.history = self.model.fit(x_train, y_train, shuffle=True, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size,
                                       verbose=2, callbacks=[checkpoint_cb, TensorB])  # , EarlyStop, ReduceLROnP
        self.model.save(self.modelpath)
        print('Continue training end!')
        print('''The trained model's weights saved successfully in ''' + self.modelweightspath)
        print('''The trained model saved successfully in ''' + self.modelpath)

    def test_model(self, x_test, y_test):  # x_test = [np.array(mri_testingdata), np.array(pet_testingdata)]
        print('Test the model:')
        test_scores = self.model.evaluate(x_test, y_test, verbose=2)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])

    def plot_metric(self, metric, savepath):
        train_metrics = self.history.history[metric]
        val_metrics = self.history.history['val_' + metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, 'bo--')
        plt.plot(epochs, val_metrics, 'ro-')
        plt.title('Training and validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, 'val_' + metric])
        plt.savefig(os.path.join(savepath, 'trainig_'+metric+'.png'))
        plt.show()

    def visualize_history(self, savepath):
        # 训练结果可视化
        self.plot_metric('loss', savepath)
        self.plot_metric('accuracy', savepath)
        
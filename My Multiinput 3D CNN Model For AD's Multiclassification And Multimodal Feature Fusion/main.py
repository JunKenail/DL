"""
作者：刘俊
日期：2022年5月10日
github地址：
"""

""" main.py（213行）
全流程主函数及相关辅助函数。
"""

import os
import fileprocess
import numpy as np
from myMultipleInput3DCNNModel import *
from niiimgshow import *
# import preprocess
import codetest
import usetrainedmodel


def get_data(filefolderpath):  # filefolderpath为存放nii格式医学图像的文件夹路径
    """ 从文件夹中读取nii格式的医学图像数据并返回[mri_data, pet_data], labels
    # 这种提取数据的方式是一次性把所有数据读入内存，数据量过大可能会由于内存不足而导致读取失败。
    # 如果数据量超过内存容量又想要在本地训练，可以分批次进行训练。
    """
    niifilelist = os.listdir(filefolderpath)
    mri_data = []
    pet_data = []
    labels = []
    for i in range(0, len(niifilelist), 2):  # 是否要重写
        niifilename_mri = niifilelist[i]
        niifilename_pet = niifilelist[i + 1]
        """ 三分类
        if 'CN' in niifilename_mri:  # 命名格式为“category_1_mri.nii”
            label = 0  # np.array([1, 0, 0])
        elif 'MCI' in niifilename_mri:
            label = 1  # np.array([0, 1, 0])
        else:  # 'AD'
            label = 2  # np.array([0, 0, 1])
        """
        #""" AD/NotAD二分类
        if 'CN' in niifilename_mri:
            label = 0
        else:
            label = 1
        #"""
        """ MCI/AD二分类
        if 'CN' in niifilename_mri:
            continue
        elif 'MCI' in niifilename_mri:
            label = 0
        else:  # 'AD'
            label = 1
        """
        mriimg = nib.load(os.path.join(filefolderpath, niifilename_mri)).get_fdata()  # 原始shape为(196,256,256)
        mriimg = np.array([np.array(mriimg, dtype=np.float32)])
        mriimg.resize((196, 256, 256))  # 统一shape
        mriimg = mriimg[:, :, :, np.newaxis]  # 在轴4上添加了大小为1的维(196, 256, 256, 1)，以便能够对数据执行3D卷积

        petimg = nib.load(os.path.join(filefolderpath, niifilename_pet)).get_fdata()  # 原始shape为(160, 160, 96, 1)，彩色图像
        petimg = np.array(petimg, dtype=np.float32)
        petimg.resize((160, 160, 96, 1))  # 统一shape

        mri_data.append(mriimg)
        pet_data.append(petimg)
        labels.append(label)

    return [np.array(mri_data), np.array(pet_data)], np.array(labels)


def main_train_on_small_sample():
    """ 小样本数据集训练全流程
    注：经过试验，我的小样本数据集大小为2GB左右，在我8G内存
    的电脑上可以运行成功，不会因内存不足而终止
    """
    print('Start:')
    # step 1. file process
    print('# Step 1. file process ↓ ↓ ↓')
    # zipfilefolderpath = '/data/cfwang/liujun_cfwang/MyData/ss0'
    # dstunzipfilefolderpath = '/data/cfwang/liujun_cfwang/MyData/ss1'
    dstdatafilefolderpath = '/data/cfwang/liujun_cfwang/MyData/ss2_test'
    # fileprocess.main(zipfilefolderpath, dstunzipfilefolderpath, dstdatafilefolderpath)  # 详细功能见fileprecess包内。
    # MRI、PET医学图像展示
    # allimgshow_func()
    print('Step 1 completed.')

    # step 2. get the training data
    print('# Step 2. get the training data ↓ ↓ ↓')
    x_train, y_train = get_data(dstdatafilefolderpath)
    print('the number of training samples: ', len(y_train))  # 训练样本个数
    print('Step 2 completed.')
    
    # step 3. build the model
    print('# Step 3. build the model ↓ ↓ ↓')
    mridata_shape, petdata_shape = (196, 256, 256, 1), (160, 160, 96, 1)  # MRI、PET输入数据的shape
    nb_classes = 3  # 做三分类
    print("the shape of MRI data  to input: ", mridata_shape)
    print("the shape of PET data  to input: ", petdata_shape)
    model = myMultipleInput3DCNNModel()  # 创建模型对象
    model.built_model(mridata_shape, petdata_shape, nb_classes)  # 构建模型
    # model.model_summary()  # 模型的详细信息
    print('Step 3 completed.')

    # step 4. trian and save the model
    print('# Step 4. trian and save the model ↓ ↓ ↓')
    model.modelpath = '/data/cfwang/liujun_cfwang/MyResults/ss_My_Multiple_Input_3DCNN_Model'
    model.modelweightspath = '/data/cfwang/liujun_cfwang/MyResults/ss_My_Multiple_Input_3DCNN_Model_best_weights.h5'
    model.log_dir = 'ss_logs'
    model.trian_and_save_model(x_train, y_train)
    print('Step 4 completed.')
   
    
    # step 5. visualize the result
    print('# step 5. visualize the result ↓ ↓ ↓')
    # resultfigsavepath = '/data/cfwang/liujun_cfwang/MyResults/ss_training_result'
    # if not os.path.isdir(resultfigsavepath):
    #     os.mkdir(resultfigsavepath)
    # model.visualize_history(resultfigsavepath)  # 这里后来使用TensorBoard对结果进行可视化
    print('Step 5 completed.')
    
    print('All steps completed.')
    print('End!')


def main_train_on_all_sample():
    """ 全样本数据集一次性训练全流程
    注：我所有样本总数据集的大小将近10GB，而我本地电脑内存只有8G，会因为内存不足而终止。
    故在本地训练只能分批次进行。后来有了服务器的硬件支持，就不用担心这个问题了。
    """
    print('Start:')
    print('# Step 1. file process ↓ ↓ ↓')
    # zipfilefolderpath = '/data/cfwang/liujun_cfwang/MyData/dt0'
    # dstunzipfilefolderpath = '/data/cfwang/liujun_cfwang/MyData/dt1'
    dstdatafilefolderpath = '/data/cfwang/liujun_cfwang/MyData/dt2'
    # fileprocess.main(zipfilefolderpath, dstunzipfilefolderpath, dstdatafilefolderpath)  # 详细功能见fileprecess包内。
    # MRI、PET医学图像展示
    # allimgshow_func()
    print('Step 1 completed.')

    print('# Step 2. get the training data ↓ ↓ ↓')
    x_train, y_train = get_data(dstdatafilefolderpath)
    print('the number of training samples: ', len(y_train))  # 训练样本个数
    print('Step 2 completed.')
    
    print('# Step 3. build the model ↓ ↓ ↓')
    mridata_shape, petdata_shape = (196, 256, 256, 1), (160, 160, 96, 1)  # MRI、PET输入数据的shape
    nb_classes = 2  # 做三分类
    print("the shape of MRI data  to input: ", mridata_shape)
    print("the shape of PET data  to input: ", petdata_shape)
    model = myMultipleInput3DCNNModel()  # 创建模型对象
    model.built_model(mridata_shape, petdata_shape, nb_classes)  # 构建模型
    # model.model_summary()  # 模型的详细信息
    print('Step 3 completed.')

    print('# Step 4. trian and save the model ↓ ↓ ↓')
    model.modelpath = '/data/cfwang/liujun_cfwang/MyResults/b1_dt_My_Multiple_Input_3DCNN_Model'
    model.modelweightspath = '/data/cfwang/liujun_cfwang/MyResults/b1_dt_My_Multiple_Input_3DCNN_Model_best_weights.h5'
    model.log_dir = 'b1_dt_logs'
    model.trian_and_save_model(x_train, y_train)
    print('Step 4 completed.')
    
    print('# step 5. visualize the result ↓ ↓ ↓')
    # resultfigsavepath = '/data/cfwang/liujun_cfwang/MyResults/dt_training_result1'
    # if not os.path.isdir(resultfigsavepath):
    #     os.mkdir(resultfigsavepath)
    # model.visualize_history(resultfigsavepath)  # 后来使用TensorBoard对结果进行可视化
    print('Step 5 completed.')
    """
    print('# Step 6. test the model ↓ ↓ ↓')
    testdatafilefolderpath = '/data/cfwang/liujun_cfwang/MyData/ss2_test'
    x_test, y_test = get_data(testdatafilefolderpath)  # 获取测试集数据
    print('the number of testing samples: ', len(y_test))  # 测试集样本个数
    model.test_model(x_test, y_test)
    print('Step 6 completed.')
    """
    print('All steps completed.')
    print('End!')


def main_train_on_batchs():
    """ 全样本数据集分批次训练全流程
    """
    data_batchfilefolderpathlist = ['/data/cfwang/liujun_cfwang/MyData/bc1', '/data/cfwang/liujun_cfwang/MyData/bc2'
                                    '/data/cfwang/liujun_cfwang/MyData/bc3', '/data/cfwang/liujun_cfwang/MyData/bc4']
    print('@ Start training on batchs:')

    print('@ Build the model ↓ ↓ ↓')
    mridata_shape, petdata_shape = (196, 256, 256, 1), (160, 160, 96, 1)  # MRI、PET输入数据的shape
    nb_classes = 3  # 做三分类
    print("the shape of MRI data  to input: ", mridata_shape)
    print("the shape of PET data  to input: ", petdata_shape)
    model = myMultipleInput3DCNNModel()
    model.built_model(mridata_shape, petdata_shape, nb_classes)
    model.modelpath = '/data/cfwang/liujun_cfwang/bc_MyResultsbc_My_Multiple_Input_3DCNN_Model'
    model.modelweightspath = '/data/cfwang/liujun_cfwang/MyResults/bc_My_Multiple_Input_3DCNN_Model_best_weights.h5'

    for i in range(len(data_batchfilefolderpathlist)):
        print('Start training on batch '+str(i+1)+':')
        dstdatafilefolderpath = data_batchfilefolderpathlist[i]
        print('# Step 1. get the training data ↓ ↓ ↓')
        x_train, y_train = get_data(dstdatafilefolderpath)
        print('the number of training samples in batch'+str(i+1)+': ', len(y_train))
        print('Step 1 completed.')

        print('# Step 2. trian and save the model ↓ ↓ ↓')
        model.log_dir = 'bc'+str(i+1)+'_logs'
        if i == 0:
            model.trian_and_save_model(x_train, y_train)
        else:
            model.trian_and_save_model_based_on_existing_model(model.modelweightspath, x_train, y_train)
        print('Step 2 completed.')

        print('# step 3. visualize and save the result ↓ ↓ ↓')
        # resultfigsavepath = '/data/cfwang/liujun_cfwang/MyResults/bc'+str(i+1)+'_training_result'
        # if not os.path.isdir(resultfigsavepath):
        #     os.mkdir(resultfigsavepath)
        # model.visualize_history(resultfigsavepath)  # 后来中使用TensorBoard对结果进行可视化
        print('Step 3 completed.')

        print('Training on batch '+str(i+1)+' end!')

    print('@ Training on batchs end!')


if __name__ == '__main__':
    main_train_on_all_sample()
    # main_train_on_small_sample()

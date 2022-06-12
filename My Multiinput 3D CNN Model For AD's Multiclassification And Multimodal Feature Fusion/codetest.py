"""
作者：刘俊
日期：2022年5月10日
github地址：
"""

""" codetest.py（56行）
此模块为代码编写过程中设计的测试代码，测试一些函数模块的准确性和功能等等
"""

import nibabel as nib
import numpy as np
from myMultipleInput3DCNNModel import *


def test_datashape():  # 测试datashape
    path = '/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/CN-M_1_mri.nii'
    img = nib.load(path)
    img_data = img.get_fdata()
    print(type(img_data))
    print(img_data.shape)
    img_data = img_data[:, :, :, np.newaxis]
    img_data = np.array(img_data, dtype=np.float32)
    print(type(img_data))
    print(img_data.shape)


def test_multiinput_model_summary():  # 显示整个模型的详细参数信息，输出结果我已经记录到了对应的“SUMMARY_....txt”文件中
    model = myMultipleInput3DCNNModel()
    mridata_shape, petdata_shape = (196, 256, 256, 1), (160, 160, 96, 1)
    model.built_model(mridata_shape, petdata_shape, 3)
    model.model_summary()


def test_denselayers_for_mri_before_fusion(nb_denseblock, filters, growth_rate, nb_layers_per_block):
    # 显示MRI端denselayers网络结构的详细参数信息，输出结果我已经记录到了对应的“SUMMARY_....txt”文件中
    mridata_shape = (196, 256, 256, 1)
    [mri_input, mri_output] = denselayers_for_mri_before_fusion(mridata_shape, nb_denseblock, filters,
                                                                growth_rate, nb_layers_per_block)
    mri_densalayers = Model(mri_input, mri_output, name='mri_denselayers')
    mri_densalayers.summary()


def test_reslayers_for_pet_before_fusion():
    # 显示PET端reslayers网络结构的详细参数信息，输出结果我已经记录到了对应的“SUMMARY_....txt”文件中
    petdata_shape = (160, 160, 96, 1)
    [pet_input, pet_output] = reslayers_for_pet_before_fusion(petdata_shape)
    pet_densalayers = Model(pet_input, pet_output, name='pet_reslayers')
    pet_densalayers.summary()


def main():
    test_datashape()
    test_denselayers_for_mri_before_fusion(4, 64, 32, [4, 6, 8, 6])
    test_reslayers_for_pet_before_fusion()
    test_multiinput_model_summary()

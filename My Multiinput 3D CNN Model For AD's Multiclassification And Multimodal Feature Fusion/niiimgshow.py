"""
作者：刘俊
日期：2022年5月10日
github地址：
"""

""" niiimgshow.py（106行）
此模块用于展示nii格式的MRI、PET图像
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def show_nii_single_img(niiimgpath, MRIorPET):
    """ 取最中间的切片以单张图片形式展示MRI或PET
    """
    img = nib.load(niiimgpath).get_fdata()
    if MRIorPET == 'MRI':
        plt.imshow(np.rot90(img[:, :, int(img.shape[2] / 2)]), cmap='gray')
    else:
        plt.imshow(np.rot90(img[:, :, int(img.shape[2] / 2), :]), cmap='jet')  # 彩色PET如何显示
        # plt.colorbar()
    plt.axis('off')
    plt.savefig('/data/cfwang/liujun_cfwang/MyResults/'+MRIorPET + '_img_show.png')
    plt.show()
# show_nii_single_img('/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_mri.nii', 'MRI')
# show_nii_single_img('/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_pet.nii', 'PET')


def show_nii_img_together(niiimgpathlist, MRIorPET):
    """ 取最中间的切片以单张图片形式同时展示CN、MCI、AD的MRI或PET，以观察比较
    Args:
        niiimgpath: [CNniiimgpath, MCIniiimgpath, ADniiimgpath]
    """
    CNimg, MCIimg, ADimg = nib.load(niiimgpathlist[0]).get_fdata(), \
                           nib.load(niiimgpathlist[1]).get_fdata(), \
                           nib.load(niiimgpathlist[2]).get_fdata()
    if MRIorPET == 'MRI':
        f, ax = plt.subplots(1, 3, figsize=(50, 30))
        ax[0].imshow(np.rot90(CNimg[:, :, int(CNimg.shape[2] / 2)]), cmap='gray')
        ax[0].axis('off')
        ax[1].imshow(np.rot90(MCIimg[:, :, int(MCIimg.shape[2] / 2)]), cmap='gray')
        ax[1].axis('off')
        ax[2].imshow(np.rot90(ADimg[:, :, int(ADimg.shape[2] / 2)]), cmap='gray')
        ax[2].axis('off')
    else:
        f, ax = plt.subplots(1, 3, figsize=(40, 30))
        ax[0].imshow(np.rot90(CNimg[:, :, int(CNimg.shape[2] / 2), :]), cmap='jet')
        ax[0].axis('off')
        ax[1].imshow(np.rot90(MCIimg[:, :, int(CNimg.shape[2] / 2), :]), cmap='jet')
        ax[1].axis('off')
        ax[2].imshow(np.rot90(ADimg[:, :, int(ADimg.shape[2] / 2), :]), cmap='jet')
        ax[2].axis('off')
    plt.savefig('/data/cfwang/liujun_cfwang/MyResults/'+MRIorPET + '_img_show_together.png')  # 这步必须在plt.show()之前
    plt.show()
# mriniiimgpathlist = ['/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/CN-M_1_mri.nii', '/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/MCI-M_1_mri.nii', '/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_mri.nii']
# show_nii_img_together(mriniiimgpathlist, 'MRI')
# petniiimgpathlist = ['/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/CN-M_1_pet.nii', '/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/MCI-M_1_pet.nii', '/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_pet.nii']
# show_nii_img_together(petniiimgpathlist, 'pet')


def show_nii_img_in_slices(niiimgpath, MRIorPET):
    """ 以切片形式展示整个MRI或PET
    """
    img = nib.load(niiimgpath).get_fdata()
    p = 0
    d = int(img.shape[2] / 40)
    if MRIorPET == 'MRI':
        f, ax = plt.subplots(4, 10, figsize=(48, 25))  # figsize过大会导致socket失效
        for i in range(4):
            for j in range(10):
                ax[i][j].imshow(np.rot90(img[:, :, p]), cmap='gray')
                ax[i][j].axis('off')
                p += d
    else:
        f, ax = plt.subplots(4, 10, figsize=(50, 20))
        for i in range(4):
            for j in range(10):
                ax[i][j].imshow(np.rot90(img[:, :, p, :]), cmap='jet')
                ax[i][j].axis('off')
                p += d
    plt.savefig('/data/cfwang/liujun_cfwang/MyResults/'+MRIorPET + '_img_show_in_slices.png')  # 这步必须在plt.show()之前
    plt.show()
# show_nii_img_in_slices('/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_mri.nii', 'MRI')
# show_nii_img_in_slices('/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_pet.nii', 'PET')


def allimgshow_func():
    """ 展示图像的代码集合
    """
    show_nii_single_img('/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_mri.nii', 'MRI')
    show_nii_single_img('/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_pet.nii', 'PET')

    mriniiimgpathlist = ['/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/CN-M_1_mri.nii',
                         '/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/MCI-M_1_mri.nii',
                         '/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_mri.nii']
    show_nii_img_together(mriniiimgpathlist, 'MRI')
    petniiimgpathlist = ['/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/CN-M_1_pet.nii',
                         '/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/MCI-M_1_pet.nii',
                         '/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_pet.nii']
    show_nii_img_together(petniiimgpathlist, 'pet')

    show_nii_img_in_slices('/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_mri.nii', 'MRI')
    show_nii_img_in_slices('/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_pet.nii', 'PET')

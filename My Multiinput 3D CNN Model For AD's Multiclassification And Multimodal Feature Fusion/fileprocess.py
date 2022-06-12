"""
作者：刘俊
日期：2022年5月10日
github地址：
"""

""" fileprocess.py（250行）
此模块包括相关的文件处理操作。
"""

import os
import shutil
from zipfile import ZipFile


def unzip_files(zipfilefolderpath, dstunzipfilefolderpath):
    """ 解压zipfilefolderpath路径下的所有.zip文件到dstunzipfilefolderpath路径下。
    """
    filelist = os.listdir(zipfilefolderpath)
    for f in filelist:
        zipfpath = os.path.join(zipfilefolderpath, f)
        if '.zip' in f:
            zipf = ZipFile(zipfpath)
            zipf.extractall(os.path.join(dstunzipfilefolderpath, f[:-4]))
            # 注意：这里使用的是绝对路径，使用绝对路径时，绝对路径长度不能超过windows系统限制的256字节
            zipf.close()


def rename_and_move_files(orgdatafilefolderpath, dstdatafilefolderpath, category):
    """ 将orgdatafilefolderpath中所有的.nii格式的pet、mri医学图像文件规范化重命名
        并在orgdatafilefolderpath整理好，同时拷贝到dstdatafilefolderpath中去。
    """
    ADNIfilepath = os.path.join(orgdatafilefolderpath, os.listdir(orgdatafilefolderpath)[0])
    objectlist = os.listdir(ADNIfilepath)
    i = 1
    for objecti in objectlist:
        imgpath = os.path.join(ADNIfilepath, objecti)
        imglist = os.listdir(imgpath)
        # 重命名pet文件
        petfilepath0 = os.path.join(imgpath, imglist[0])
        petfilepath1 = os.path.join(petfilepath0, os.listdir(petfilepath0)[0])
        petfilepath2 = os.path.join(petfilepath1, os.listdir(petfilepath1)[0])
        petfilepathfinal = os.path.join(petfilepath2, os.listdir(petfilepath2)[0])
        # print(petfilepathfinal)
        petnewpathname = os.path.join(petfilepath2, category + '_' + str(i) + '_pet.nii')  # 规范化命名
        os.rename(petfilepathfinal, petnewpathname)
        # 重命名mri文件
        mrifilepath0 = os.path.join(imgpath, imglist[1])
        mrifilepath1 = os.path.join(mrifilepath0, os.listdir(mrifilepath0)[0])
        mrifilepath2 = os.path.join(mrifilepath1, os.listdir(mrifilepath1)[0])
        mrifilepathfinal = os.path.join(mrifilepath2, os.listdir(mrifilepath2)[0])
        # print(mrifilepathfinal)
        mrinewpathname = os.path.join(mrifilepath2, category + '_' + str(i) + '_mri.nii')
        os.rename(mrifilepathfinal, mrinewpathname)
        # 复制、移动文件
        shutil.copy(petnewpathname, dstdatafilefolderpath)
        shutil.copy(mrinewpathname, dstdatafilefolderpath)
        shutil.move(petnewpathname, orgdatafilefolderpath)
        shutil.move(mrinewpathname, orgdatafilefolderpath)
        i += 1
    shutil.rmtree(ADNIfilepath)


def main(zipfilefolderpath, dstunzipfilefolderpath, dstdatafilefolderpath):
    """ 将从ADNI中下载到的.zip源数据文件进行解压，
        并对其内.nii格式的医学图像数据进行规范化重命名，
        并将其移动到同一文件夹dstdatafilefolderpath内
    """
    unzip_files(zipfilefolderpath, dstunzipfilefolderpath)  # 1
    filelist = os.listdir(dstunzipfilefolderpath)
    for f in filelist:
        file = os.path.join(dstunzipfilefolderpath, f)
        rename_and_move_files(file, dstdatafilefolderpath, f)
        
# main(r"D:\lj\MyData\ss0", r"D:\lj\MyData\ss1", r"D:\lj\MyData\ss2_test")
""" #具体说明#
上面几个函数涉及的参数关系（以我的“ss0”、“ss1”、“ss2”这三个数据文件为例）：
（注意：这里使用的都是绝对路径）

刚开始，“ss0”、“ss1”、“ss2”存放在“MyData”文件件内，
“ss0”文件夹内存放着从ADNI中下载的源数据zip压缩文件，“ss1”、“ss2”为空。即：
MyData
{  
ss0{
    AD-F.zip  # 以“category-gender”规范化命名
    AD-M.zip
    CN-F.zip
    CN-M.zip
    MCI-F.zip
    MCI-M.zip
}
ss1{}
ss2{}
}
    
######
def unzip_files(zipfilefolderpath, dstunzipfilefolderpath):
    zipfilefolderpath是“ss0”文件夹的绝对路径，即“...\MyData\ss0”；
    dstunzipfilefolderpath是“ss1”文件夹的绝对路径；
    这个函数的功能是把“ss0”文件夹（zipfilefolderpath）内的zip压缩文件解压到“ss1”文件夹（dstunzipfilefolderpath）内。

step 1：首先执行这个函数，也就是def main()里的第1步，执行之后的效果是：
MyData
{
ss0{
    AD-F.zip
    AD-M.zip
    CN-F.zip
    CN-M.zip
    MCI-F.zip
    MCI-M.zip
}
ss1{
    AD-F{  # 文件夹里面的文件组织有点复杂，以解压后的“AD-F”为例，其内部文件组织如下
        ADNI{
            016_S_4353{
                MT1__GradWarp__N3m{
                    2012-03-31_11_06_55.0{
                        S145696{
                            ADNI_016_S_4353_MR_MT1__GradWarp__N3m_Br_20120416133844478_S145696_I297620.nii
                            # 这是我们要的MRI数据文件
                            # 之后会对它进行规范化重命名，会被重命名为“AD-F_1_mri.nii”
                        }
                    }
                }
                Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution{
                    2012-01-31_10_24_06.0{
                        I282615{
                            ADNI_016_S_4353_PT_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution_Br_20120203154609620_8_S138936_I282615.nii
                            # 这是我们要的PET数据文件
                            # 之后会对它进行规范化重命名，会被重命名为“AD-F_1_pet.nii”
                        }
                    }
                }
            }
            011_S_4949{
                MT1__GradWarp__N3m{
                    2012-09-19_09_59_04.0{
                        S168025{
                            ADNI_011_S_4949_MR_MT1__GradWarp__N3m_Br_20121001123801995_S168025_I337436.nii
                            # 这是我们要的MRI数据文件
                            # 之后会对它进行规范化重命名，会被重命名为“AD-F_2_mri.nii”
                        }
                    }
                }
                Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution{
                    2012-10-10_12_12_01.0{
                        I340490{
                            ADNI_011_S_4949_PT_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution_Br_20121016170612002_57_S170416_I340490.nii
                            # 这是我们要的PET数据文件
                            # 之后会对它进行规范化重命名，会被重命名为“AD-F_2_mri.nii”
                        }
                    }
                }
            }
            011_S_4906
                .
                .
                .
        }
    }
    AD-M
    CN-F
    CN-M
    MCI-F
    MCI-M
}
ss2{}
}

######
def rename_and_move_files(orgdatafilefolderpath, dstdatafilefolderpath, category)：
    orgdatafilefolderpath是“ss1”文件夹中的文件（如“AD-F”）的绝对路径，即“...\MyData\ss1\AD-F”
    dstdatafilefolderpath是“ss2”文件夹的绝对路径；
    category即类别，用于规范化重命名，若orgdatafilefolderpath为“...\MyData\ss1\AD-F”，则category则为“AD-F”，
也即“ss1”文件夹中的文件的名字。
    这个函数的操作有两个，一个遍历处理“ss1”文件夹中的文件，以其内的“AD-F”文件夹为例，会规范化重命名里面我们想要的.nii
数据文件，并重新整理“AD-F”文件夹里的文件；二是，把规范化重命名后我们想要的的.nii数据文件同时复制一份统一放到“ss2”文件夹
（dstdatafilefolderpath）内。

step 2：然后遍历“ss2”文件夹（dstdatafilefolderpath）内的各个解压后的文件，对每个解压后的文件依次执行这个函数，执行完
效果如下，这也是执行完文件处理fileprocess包内def main()函数之后的效果：
MyData
{
ss0{
    AD-F.zip
    AD-M.zip
    CN-F.zip
    CN-M.zip
    MCI-F.zip
    MCI-M.zip
}
ss1{
    AD-F{  # 同样“AD-F”为例，其内部文件组织变成了：
        AD-F_1_mri.nii  # 把我们想要的文件规范化重命名并直接放到了这里
        AD-F_1_pet.nii  # 规范化重命名格式为“category_number_modality.nii”
        AD-F_2_mri.nii
        AD-F_2_pet.nii
             .
             .
             .
        }
    }
    AD-M{  # 同样的
        AD-M_1_mri.nii
        AD-M_1_pet.nii
        AD-M_2_mri.nii
        AD-M_2_pet.nii
             .
             .
             .
    }
    CN-F
    CN-M
    MCI-F
    MCI-M
}
ss2{  # 把所有规范化命名的数据文件统一放到一起。而我们的数据集就是来自于这里！
    AD-F_1_mri.nii
    AD-F_1_pet.nii
    AD-F_2_mri.nii
    AD-F_2_pet.nii
         .
         .
         .
    AD-M_1_mri.nii
    AD-M_1_pet.nii
    AD-M_2_mri.nii
    AD-M_2_pet.nii
         .
         .
         .
    CN-F_1_mri.nii
    CN-F_1_pet.nii
         .
         .
         .     
         .
         .
         .
}
}

def main(zipfilefolderpath, dstunzipfilefolderpath, dstdatafilefolderpath):
    综上：有了这个fileprocess.main()，先创建zipfilefolderpath（“ss0”）, dstunzipfilefolderpath（“ss1”）, 
dstdatafilefolderpath（“ss2”）这三个文件夹，然后把从ADNI下载到的规范化命名的zip压缩的原数据文件放进zipfilefold
erpath文件夹内，再执行fileprocess.main(zipfilefolderpath, dstunzipfilefolderpath, dstdatafilefolderpath)
，就能得到dstunzipfilefolderpath（“ss1”）、dstdatafilefolderpath（“ss2”）的文件内容！

"""


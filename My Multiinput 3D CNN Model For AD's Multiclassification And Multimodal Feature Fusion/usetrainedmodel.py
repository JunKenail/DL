"""
作者：刘俊
日期：2022年5月10日
github地址：
"""

""" usetrainedmodel.py（68行）
此模块为使用已有或训练好的模型的一些操作
"""

from keras.models import load_model
from keras import callbacks
import numpy as np

# 改成读文件预测
def predict_samples(existingmodelpath, x_input):  # x_input为np.array类型
    """
    # 注意x_input的格式：
    x_input的格式同main.py模块中get_data()函数的第一个返回值。
    # 注意x_input中MRI、PET数据的shape
    x_input = [np.array(mridata)_with_shape_of_(196, 256, 256, 1), np.array(petdata)_with_shape_of_(160, 160, 96, 1)]
    # 此函数的使用注意见line70
    """
    model = load_model(existingmodelpath)
    y_prob = model.predict(x_input)
    y_category = np.argmax(y_prob, axis=1)
    print("Prediction results ↓ ↓ ↓")
    print("Probability of each category for each sample:")
    print(y_prob)
    print("Category of each sample(0-CN, 1-MCI, 2-AD):")
    print(y_category)


def test_on_existing_model(existingmodelh5path, x_test, y_test):  # x_test、y_test为np.array类型
    # 使用测试集测试模型
    model = load_model(existingmodelh5path)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])


def trian_and_save_model_based_on_existing_model(existingmodelh5path, x_train, y_train):  # x_train、y_train为np.array类型
    print('Continue training begins:')
    model = load_model(existingmodelh5path)
    checkpoint_cb = callbacks.ModelCheckpoint(existingmodelh5path, monitor='loss',
                                              save_best_only=True, mode='min')
    # ReduceLROnP = callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=10e-6)  # 动态调整学习率
    TensorB = callbacks.TensorBoard(log_dir='continue_logs')  # 使用TensorBoard对结果进行可视化。
    history = model.fit(x_train, y_train, shuffle=True, validation_split=0.2, epochs=16, batch_size=8,
                        verbose=2, callbacks=[checkpoint_cb, TensorB])
    print('Continue training end!')
    print('''The trained model's weights saved successfully!''')


""" def predict_samples(existingmodelh5path, x_input)函数的使用说明： """
# # # 预测单个样本的代码示例：
# mridt = np.array(nib.load('/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_mri.nii').get_fdata())
# mridt.resize((196, 256, 256))  # 统一shape
# mridt = mridt[:, :, :, np.newaxis]
# petdt = np.array(nib.load('/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing/AD-M_1_pet.nii').get_fdata())
# petdt.resize((160, 160, 96, 1))  # 统一shape
# existingmodelh5path = '/data/cfwang/liujun_cfwang/MyCodes/dt_My_Multiple_Input_3DCNN_Model_best_weights.h5'
# usetrainedmodel.predict_samples(existingmodelh5path, [np.array([mridt]), np.array([petdt])])

# # # 预测多个样本，建议先用main.get_data()得到x_input
# x_input, y_null = get_data('/data/cfwang/liujun_cfwang/MyData/MRI_PET_img_for_showing')
# existingmodelh5path = '/data/cfwang/liujun_cfwang/MyCodes/dt_My_Multiple_Input_3DCNN_Model_best_weights.h5'
# usetrainedmodel.predict_samples(existingmodelh5path, x_input)


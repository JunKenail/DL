from usetrainedmodel import *
from main import *

print('Get the testing data ↓ ↓ ↓')
x_test, y_test = get_data("/data/cfwang/liujun_cfwang/MyData/ss2_test/")
print('Get successfully!')  # 训练样本个数
print('The number of testing samples: ', len(y_test))  # 训练样本个数
print('Do testing ↓ ↓ ↓')
existingmodelh5path = "/data/cfwang/liujun_cfwang/MyResults/dt_My_Multiple_Input_3DCNN_Model3"
test_on_existing_model(existingmodelh5path, x_test, y_test)




这是在MobaXterm上使用服务器运行的代码。
如果要在本地训练，只需改改代码里的绝对路径就行。

修改模型之前：
517dt_main1.log  动态学习率（0.005），batch_size=8
517dt_main2.log  非动态学习率（0.001），batch_size=8
517dt_main3.log  动态学习率（0.01），batch_size=4

修改模型后：
518dt_main1.log  动态学习率（0.005），batch_size=8 ，epochs=16  17033
518dt_main2.log  动态学习率（0.005），batch_size=16 ，epochs=16  17668

大模型：
main3.log  非动态学习率（0.0005），batch_size=8 ，epochs=16  28693 kill
main4.log  非动态学习率（0.001），batch_size=8 ，epochs=16  29214 kill
main5.log  动态学习率（0.01→1/4），batch_size=8 ，epochs=16 

小模型：
小样本
ss_main1.log 动态学习率（0.001→1/4），batch_size=4 ，epochs=16  36163 kill
ss_main2/3.log 非动态学习率（0.001），batch_size=4 ，epochs=20  11741 47068
ss_main4.log 动态学习率（0.005→1/2），batch_size=4 ，epochs=20  56616
ss_main5.log  默认adam，batch_size=4 ，epochs=24  38620
ss_main6.log  默认nadam，batch_size=4 ，epochs=24  35290
大样本 
dt_main12.log 非动态学习率（0.001），validation_split=0.2，batch_size=16 ，epochs=16，持续学习  36641 60655

dt_main.log  默认adam，batch_size=36 ，epochs=8, with dropout  79913
dt_main2.log  默认adam，batch_size=36 ，epochs=8, with BN  27545
dt_main3.log  small model， 默认adam，动态学习率（0.001→0.8），batch_size=50 ，epochs=16, with dropout  1342

二分类再二分类
b_dt_main1.log 默认adam，LeakyReLU, 动态学习率（0.001→0.8），batch_size=50 ，epochs=8, with BN 2086
b_dt_main2.log 默认adam，LeakyReLU, 动态学习率（0.001→0.8），batch_size=50 ，epochs=8, with BN kill
b1_dt_main.log 默认adam，LeakyReLU, 动态学习率（0.001→0.8），batch_size=50 ，epochs=8, with BN 6671
b2_dt_main.log 默认adam，LeakyReLU, 动态学习率（0.001→0.8），batch_size=50 ，epochs=8, with BN 5367
import numpy as np
import matplotlib.pyplot as plt
f=open('./train_vanishing_points-2017-09-01-22-03-02.log')
lines=f.readlines()
Test_Loss_data=[]
Train_Loss_data=[]
for line in lines:
    Train_Loss_index=line.find('Train_Loss')
    if Train_Loss_index is -1:  # the line is test loss
        Test_Loss_index = line.find('Test_Loss')
        print(Test_Loss_index)
        Test_Loss=line[Test_Loss_index+11:Test_Loss_index+17]
        Test_Loss=float(Test_Loss)
        Test_Loss_data.append(Test_Loss)
    else:
        print(Train_Loss_index)
        Train_Loss=line[Train_Loss_index+12:Train_Loss_index+18]
        Train_Loss=float(Train_Loss)
        Train_Loss_data.append(Train_Loss)
print(Test_Loss_data)
print(Train_Loss_data)
    # loss=line[Loss_index+6:Loss_index+12]
    # loss=float(loss)
    # data.append(loss)
test_data_len=len(Test_Loss_data)
Test_Loss_data=np.array(Test_Loss_data)
index=np.array(range(test_data_len))
tmp_data = Test_Loss_data.reshape((93, 937))
# print(tmp_data.shape)
mean_data=np.mean(tmp_data,axis=1)
mean_data_len=np.array(range(len(mean_data)))
plt.figure(1)
plt.plot(mean_data_len, mean_data,"r-",label='test_loss')


Train_Loss_data=np.array(Train_Loss_data)
tmp_train_data=Train_Loss_data.reshape((93, 937))
mean_train_data=np.mean(tmp_train_data,axis=1)
plt.plot(mean_data_len, mean_train_data,"g-",label="train_loss")
plt.title('resnet_finetune0901')
plt.legend()  # show the title of the diagram
plt.show()

'''
f=open('/home/jensen/Nutstore/distortion_correction/plot_curve/distortion_correction_gaussian_alexnet_no_finetune.txt')
lines=f.readlines()
data=[]
for line in lines:
    Loss_index=line.find('Loss')
    loss=line[Loss_index+6:Loss_index+12]
    loss=float(loss)
    data.append(loss)
data_len=len(data)
data=np.array(data)
index=np.array(range(data_len))
# print(data.shape)
tmp_data = data.reshape((50, 191))
# print(tmp_data.shape)
mean_data=np.mean(tmp_data,axis=1)
mean_data_len=np.array(range(len(mean_data)))
plt.figure(1)
plt.plot(mean_data_len, mean_data)
plt.title('distortion_using_gaussian_filter_with_alexnet')
plt.show()
'''
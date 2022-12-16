# image classification
import numpy as np
import cv2


#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
import math


class Classifier(object):
    def __init__(self, filePath):
        self.filePath = filePath

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_data(self):
        TrainData = []
        TestData = []
        num_in_test = 1000 # test image number
        data = self.unpickle(self.filePath)
        train = np.reshape(data[b'data'][0:2000,:], (2000, 3, 32 * 32))
        labels = np.reshape(data[b'labels'][0:2000], (2000, 1))
        num=0
        for i in range(len(labels)):
            if labels[i] != 0:
                num+=1
                labels[i] = 1  # 等于1说明不是飞机
        #print('train: negative_sample:{0} positive_sample:{1}'.format(num, len(labels) - num))
        fileNames = np.reshape(data[b'filenames'][0:2000], (2000, 1))
        datalebels = zip(train, labels, fileNames) # turn it in tuple
        TrainData.extend(datalebels)
        test = np.reshape(data[b'data'][2000:2000+num_in_test,:], (num_in_test, 3, 32 * 32))
        labels = np.reshape(data[b'labels'][2000:2000+num_in_test], (num_in_test, 1))
        num=0
        for i in range(len(labels)):
            if labels[i] != 0:
                num+=1
                labels[i] = 1  # 等于1说明不是飞机
        #print('test: negative_sample:{0} positive_sample:{1}'.format(num,len(labels)-num))
        fileNames = np.reshape(data[b'filenames'][2000:2000+num_in_test], (num_in_test, 1))
        TestData.extend(zip(test, labels, fileNames))
        print("data read finished!")
        return TrainData, TestData,test

    def get_hog_feat(self, image):
        winSize=(32,32)
        blockSize=(4,4)
        blockStride=(1,1)
        cellSize=(2,2)
        Bin=9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, Bin)
        hist=hog.compute(image,(8,8))
        return hist.reshape(-1)
    def get_feat(self, TrainData, TestData,a):
        train_feat = []
        test_feat = []
        for data in TestData:
            image = np.reshape(data[0].T, (32, 32, 3))
            if a!='1':
                #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
                fd = self.get_hog_feat(image)
            else:
                img = image.flatten()  # 对图像进行降维操作，方便算法计算
                fd = img / np.mean(img)  # 归一化，突出特征
            fd = np.concatenate((fd, data[1]))
            test_feat.append(fd)
        test_feat = np.array(test_feat,dtype='float32')
        print("Test features are extracted")
        for data in TrainData:
            image = np.reshape(data[0].T, (32, 32, 3))
            if a!='1':
                fd = self.get_hog_feat(image)
                #print('ok')
            else:
                img = image.flatten()  # 对图像进行降维操作，方便算法计算
                fd = img / np.mean(img)  # 归一化，突出特征
            fd = np.concatenate((fd, data[1]))
            train_feat.append(fd)
        train_feat = np.array(train_feat,dtype='float32')
        print("Train features are extracted")
        return train_feat, test_feat

    def classification(self, train_feat, test_feat,test,a):
        if a=='1':
            name='RGB'
        else:
            name='HOG'
        svm = cv2.ml.SVM_create()  # 创建svm学习模型
        svm.setType(cv2.ml.SVM_C_SVC)  # 类型为svm分类
        svm.setKernel(cv2.ml.SVM_LINEAR)  # 设置svm的内核为线性分类器
        svm.setC(0.01)
        train_label=[]
        print("Training a Linear SVM Classifier.")
        for i in range(train_feat.shape[0]):
            train_label1=[int(train_feat[i,-1])]
            train_label.append(train_label1)
        train_label=np.array(train_label,dtype='int')
        print(train_label.shape)
        print(train_feat[:,:-1].shape)

        svm.train(train_feat[:, :-1],cv2.ml.ROW_SAMPLE,train_label) #最后一位是标志
        predict_result=svm.predict(test_feat[:,:-1])
        predict_result=predict_result[1].flatten()
        print(predict_result.shape)
        #score,predict_result = clf.score(test_feat[:, :-1],test_feat[:,-1],predict_result)
        print('This is prediction results:\n')
        for i in range(len(predict_result)):
            print(int(predict_result[i]),end="")
        print("\n")
        print('This is the test labels:\n')
        for i in range(len(predict_result)):
            print(int(test_feat[i, -1]),end="")
        print("\n")
        num = 0
        plane=[]
        count_true=0
        not_plane=[]
        count_false=0
        for i in range(len(predict_result)):
            if int(predict_result[i]) == int(test_feat[i, -1]):
                num += 1
                print(i,end=" ")

                if predict_result[i]==0 and count_true<5:
                    plane.append(i)
                    count_true=count_true+1
                elif count_false<5:
                    not_plane.append(i)
                    count_false += 1
        print('correct prediction numbers:\n'+str(num))
        #print(plane)
        print('The classification accuracy is {0}'.format(num/len(predict_result)))
        count=0
        for i in plane:
            count+=1
            img1_r = test[i][0,:].reshape(32,-1)
            img1_g = test[i][1,:].reshape(32,-1)
            img1_b = test[i][2,:].reshape(32,-1)
            img = cv2.merge([img1_r, img1_g, img1_b])
            cv2.imshow('plane', img.astype('uint8'))
            cv2.imwrite('../results/{0}_plane{1}.jpg'.format(name,count),img)
            cv2.waitKey()
        count=0
        for i in not_plane:
            count+=1
            img1_r = test[i][0, :].reshape(32, -1)
            img1_g = test[i][1, :].reshape(32, -1)
            img1_b = test[i][2, :].reshape(32, -1)
            img = cv2.merge([img1_r, img1_g, img1_b])
            cv2.imshow('not_plane', img.astype('uint8'))
            cv2.imwrite('../results/{0}_not_plane{1}.jpg'.format(name,count), img)
            cv2.waitKey()
    def run(self,a):
        TrainData, TestData,test = self.get_data()
        train_feat, test_feat = self.get_feat(TrainData, TestData,a)
        self.classification(train_feat, test_feat,test,a)




if __name__ == '__main__':
    test_true=[]
    filePath='./cifar-10/cifar-10-batches-py/data_batch_1'
    cf = Classifier(filePath)
    a=input("RGB feature press 1,HOG feature press 2:")
    cf.run(a)







##########################################################################################
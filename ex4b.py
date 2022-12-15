# image classification
import numpy as np
import cv2


#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
import math
from sklearn.svm import LinearSVC

class SVM:
    def __init__(self, max_iter=100, kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel

    #参数初始化
    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        self.alpha = np.ones(self.m)
        self.computer_product_matrix()#为了加快训练速度创建一个内积矩阵
        # 松弛变量
        self.C = 1.0
        # 将Ei保存在一个列表里
        self.create_E()

    #KKT条件判断
    def judge_KKT(self, i):
        y_g = self.function_g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    #计算内积矩阵#如果数据量较大，可以使用系数矩阵
    def computer_product_matrix(self):
        self.product_matrix = np.zeros((self.m,self.m)).astype(float)
        for i in range(self.m):
            for j in range(self.m):
                if self.product_matrix[i][j]==0.0:
                    self.product_matrix[i][j]=self.product_matrix[j][i]= self.kernel(self.X[i], self.X[j])

    # 核函数
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return np.dot(x1,x2)
        elif self._kernel == 'poly':
            return (np.dot(x1,x2) + 1) ** 2
        return 0

    #将Ei保存在一个列表里
    def create_E(self):
        self.E=(np.dot((self.alpha * self.Y),self.product_matrix)+self.b)-self.Y

    # 预测函数g(x)
    def function_g(self, i):
        return self.b+np.dot((self.alpha * self.Y),self.product_matrix[i])

    #选择变量
    def select_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        for i in index_list:
            if self.judge_KKT(i):
                continue
            E1 = self.E[i]
            # 如果E2是+，选择最小的；如果E2是负的，选择最大的
            if E1 >= 0:
                j =np.argmin(self.E)
            else:
                j = np.argmax(self.E)
            return i, j

    #剪切
    def clip_alpha(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha
    #训练函数，使用SMO算法
    def Train(self, features, labels):
        self.init_args(features, labels)
        #SMO算法训练
        for t in range(self.max_iter):
            i1, i2 = self.select_alpha()

            # 边界
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta=K11+K22-2K12
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2 * self.kernel(
                self.X[i1], self.X[i2])
            if eta <= 0:
                # print('eta <= 0')
                continue

            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta  # 此处有修改，根据书上应该是E1 - E2，书上130-131页
            alpha2_new = self.clip_alpha(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.create_E()

    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])

        return 1 if r > 0 else 0

    def score(self, X_test, y_test,predict_result):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            predict_result.append(result)
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test),predict_result

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
        num_in_test = 200  # test image number
        data = self.unpickle(self.filePath)
        train = np.reshape(data[b'data'][0:2000,:], (2000, 3, 32 * 32))
        labels = np.reshape(data[b'labels'][0:2000], (2000, 1))
        for i in range(len(labels)):
            if labels[i] != 0:
                labels[i] = 1  # 等于1说明不是飞机
        fileNames = np.reshape(data[b'filenames'][0:2000], (2000, 1))
        datalebels = zip(train, labels, fileNames) # turn it in tuple
        TrainData.extend(datalebels)
        test = np.reshape(data[b'data'][2000:2000+num_in_test,:], (num_in_test, 3, 32 * 32))
        labels = np.reshape(data[b'labels'][2000:2000+num_in_test], (num_in_test, 1))
        for i in range(len(labels)):
            if labels[i] != 0:
                labels[i] = 1  # 等于1说明不是飞机
        fileNames = np.reshape(data[b'filenames'][2000:2000+num_in_test], (num_in_test, 1))
        TestData.extend(zip(test, labels, fileNames))
        print("data read finished!")
        return TrainData, TestData,test

    def get_hog_feat(self, image, stride=8, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        cx, cy = pixels_per_cell
        bx, by = cells_per_block
        sx, sy = image.shape
        n_cellsx = int(np.floor(sx // cx))  # number of cells in x
        n_cellsy = int(np.floor(sy // cy))  # number of cells in y
        n_blocksx = (n_cellsx - bx) + 1
        n_blocksy = (n_cellsy - by) + 1
        gx = np.zeros((sx, sy), dtype=np.float32)
        gy = np.zeros((sx, sy), dtype=np.float32)
        eps = 1e-5
        grad = np.zeros((sx, sy, 2), dtype=np.float32)
        for i in range(1, sx-1):
            for j in range(1, sy-1):
                gx[i, j] = image[i, j-1] - image[i, j+1]
                gy[i, j] = image[i+1, j] - image[i-1, j]
                grad[i, j, 0] = np.arctan(gy[i, j] / (gx[i, j] + eps)) * 180 / math.pi
                if gx[i, j] < 0:
                    grad[i, j, 0] += 180
                grad[i, j, 0] = (grad[i, j, 0] + 360) % 360
                grad[i, j, 1] = np.sqrt(gy[i, j] ** 2 + gx[i, j] ** 2)
        normalised_blocks = np.zeros((n_blocksy, n_blocksx, by * bx * orientations))
        for y in range(n_blocksy):
            for x in range(n_blocksx):
                block = grad[y*stride:y*stride+16, x*stride:x*stride+16]
                hist_block = np.zeros(32, dtype=np.float32)
                eps = 1e-5
                for k in range(by):
                    for m in range(bx):
                        cell = block[k*8:(k+1)*8, m*8:(m+1)*8]
                        hist_cell = np.zeros(8, dtype=np.float32)
                        for i in range(cy):
                            for j in range(cx):
                                n = int(cell[i, j, 0] / 45)
                                hist_cell[n] += cell[i, j, 1]
                        hist_block[(k * bx + m) * orientations:(k * bx + m + 1) * orientations] = hist_cell[:]
                normalised_blocks[y, x, :] = hist_block / np.sqrt(hist_block.sum() ** 2 + eps)
        #print(normalised_blocks.ravel().shape)
        return normalised_blocks.ravel() #展成1维向量

    def get_feat(self, TrainData, TestData,a):
        train_feat = []
        test_feat = []
        for data in TestData:
            image = np.reshape(data[0].T, (32, 32, 3))

            if a!='1':
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
                fd = self.get_hog_feat(gray)
            else:
                img = image.flatten()  # 对图像进行降维操作，方便算法计算
                fd = img / np.mean(img)  # 归一化，突出特征
            fd = np.concatenate((fd, data[1]))
            test_feat.append(fd)
        test_feat = np.array(test_feat)
        print("Test features are extracted")
        for data in TrainData:
            image = np.reshape(data[0].T, (32, 32, 3))
            if a!='1':
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
                fd = self.get_hog_feat(gray)
            else:
                img = image.flatten()  # 对图像进行降维操作，方便算法计算
                fd = img / np.mean(img)  # 归一化，突出特征
            fd = np.concatenate((fd, data[1]))
            train_feat.append(fd)
        train_feat = np.array(train_feat)
        print("Train features are extracted")
        return train_feat, test_feat

    def classification(self, train_feat, test_feat,test,a):
        if a=='1':
            name='RGB'
        else:
            name='HOG'
        clf = SVM(max_iter=200)
        print("Training a Linear SVM Classifier.")
        clf.Train(train_feat[:, :-1], train_feat[:, -1]) #最后一位是标志
        predict_result=[]
        score,predict_result = clf.score(test_feat[:, :-1],test_feat[:,-1],predict_result)

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
                #print(i,end=" ")

                if predict_result[i]==0 and count_true<5:
                    plane.append(i)
                    count_true=count_true+1
                elif count_false<5:
                    not_plane.append(i)
                    count_false += 1
        print('correct prediction numbers:\n'+str(num))
        rate = score
        #print(plane)
        print('The classification accuracy is %f' % rate)
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
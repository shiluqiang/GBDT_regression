# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:28:28 2018

@author: lj
"""
import CART_regression_tree
import numpy as np

def load_data(data_file):
    '''导入训练数据
    input:  data_file(string):保存训练数据的文件
    output: data(list):训练数据
    '''
    data_X = []
    data_Y = []
    f = open(data_file)
    for line in f.readlines():
        sample = []
        lines = line.strip().split("\t")
        data_Y.append(float(lines[-1]))
        for i in range(len(lines) - 1):
            sample.append(float(lines[i]))  # 转换成float格式
        data_X.append(sample)
    f.close()    
    return data_X,data_Y

class GBDT_RT(object):
    '''GBDT回归算法类
    '''
    def __init__(self):
        self.trees = None ##用于存放GBDT的树
        self.learn_rate = learn_rate ## 学习率，防止过拟合
        self.init_value = None ##初始数值
        self.fn = lambda x: x
        
    def get_init_value(self,y):
        '''计算初始数值为平均值
        input:y(list):样本标签列表
        output:average(float):样本标签的平均值
        '''
        average = sum(y)/len(y)
        return average
    
    def get_residuals(self,y,y_hat):
        '''计算样本标签标签与预测列表的残差
        input:y(list):样本标签列表
              y_hat(list):预测标签列表
        output:y_residuals(list):样本标签标签与预测列表的残差
        '''
        y_residuals = []
        for i in range(len(y)):
            y_residuals.append(y[i] - y_hat[i])
        return y_residuals
    
    def fit(self,data_X,data_Y,n_estimators,learn_rate,min_sample, min_err):
        '''训练GBDT模型
        input:self(object):GBDT_RT类
              data_X(list):样本特征
              data_Y(list):样本标签
              n_estimators(int):GBDT中CART树的个数
              learn_rate(float):学习率
              min_sample(int):学习CART时叶节点的最小样本数
              min_err(float):学习CART时最小方差
        '''
        ## 初始化预测标签和残差
        self.init_value = self.get_init_value(data_Y)
        
        n = len(data_Y)
        y_hat = [self.init_value] * n ##初始化预测标签
        y_residuals = self.get_residuals(data_Y,y_hat)
        
        self.trees = []
        self.learn_rate = learn_rate
        ## 迭代训练GBDT
        for j in range(n_estimators):
            idx = range(n)
            X_sub = [data_X[i] for i in idx] ## 样本特征列表
            residuals_sub = [y_residuals[i] for i in idx] ## 标签残差列表
            
            tree = CART_regression_tree.CART_RT(X_sub,residuals_sub, min_sample, min_err).fit()
            res_hat = [] ##残差的预测值
            for m in range(n):
                res_hat.append(CART_regression_tree.predict(data_X[m],tree))
            ## 计算此时的预测值等于原预测值加残差预测值
            y_hat = [y_hat[i] + self.learn_rate * res_hat[i] for i in idx]
            y_residuals = self.get_residuals(data_Y,y_hat)
            self.trees.append(tree)
            
    def GBDT_predict(self,xi):
        '''预测一个样本
        '''
        return self.fn(self.init_value + sum(self.learn_rate * CART_regression_tree.predict(xi,tree) for tree in self.trees))
    
    def GBDT_predicts(self,X):
        '''预测多个样本
        '''
        return [self.GBDT_predict(xi) for xi in X]

def error(Y_test,predict_results):
    '''计算预测误差
    input:Y_test(list):测试样本标签
          predict_results(list):测试样本预测值
    output:error(float):均方误差
    '''
    Y = np.mat(Y_test)
    results = np.mat(predict_results)
    error = np.square(Y - results).sum() / len(Y_test)
    return error


if __name__ == '__main__':
    print ("------------- 1.load data ----------------")
    X_data,Y_data = load_data("sine.txt") 
    X_train = X_data[0:150]
    Y_train = Y_data[0:150]
    X_test = X_data[150:200]
    Y_test = Y_data[150:200]
    print('------------2.Parameters Setting-----------')
    n_estimators = 4
    learn_rate = 0.5
    min_sample = 30
    min_err = 0.3
    print ("--------------3.build GBDT ---------------")
    gbdt_rt = GBDT_RT()
    gbdt_rt.fit(X_train,Y_train,n_estimators,learn_rate,min_sample, min_err)
    print('-------------4.Predict Result--------------')
    predict_results = gbdt_rt.GBDT_predicts(X_test)
    print('--------------5.Predict Error--------------')
    error = error(Y_test,predict_results)
    print('Predict error is: ',error)
    
    

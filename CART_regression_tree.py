# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:25:03 2018

@author: lj
"""
import numpy as np
import copy

class node:
    '''树的节点的类
    '''
    def __init__(self, fea=-1, value=None, results=None, right=None, left=None):
        self.fea = fea  # 用于切分数据集的属性的列索引值
        self.value = value  # 设置划分的值
        self.results = results  # 存储叶节点的值
        self.right = right  # 右子树
        self.left = left  # 左子树

class CART_RT(object):
    '''CART算法的类
    '''
    def __init__(self,data_X,data_Y,min_sample, min_err):
        '''初始化CART类参数
        '''
        self.data_X = data_X #待回归样本数据的特征
        self.data_Y = data_Y #待回归样本数据的标签
        self.min_sample = min_sample # 每个叶节点最多的样本数
        self.min_err = min_err #最小方差
        
    def fit(self):
        '''构建树
            input:  data(list):训练样本
                    min_sample(int):叶子节点中最少的样本数
                    min_err(float):最小的error
            output: node:树的根结点
        '''  
        # 将样本特征与样本标签合成完整的样本
        data = combine(self.data_X,self.data_Y)
        # 构建决策树，函数返回该决策树的根节点
        if len(data) <= self.min_sample:
            return node(results=leaf(data))
    
        # 1、初始化
        best_err = err_cnt(data)
        bestCriteria = None  # 存储最佳切分属性以及最佳切分点
        bestSets = None  # 存储切分后的两个数据集
    
        # 2、开始构建CART回归树
        feature_num = len(data[0]) - 1
        for fea in range(0, feature_num):
            feature_values = {}
            for sample in data:
                feature_values[sample[fea]] = 1
            for value in feature_values.keys():
                # 2.1、尝试划分
                (set_1, set_2) = split_tree(data, fea, value)
                combine_set_1 = combine(set_1[0],set_1[1])
                combine_set_2 = combine(set_2[0],set_2[1])
                if len(combine_set_1) < 2 or len(combine_set_2) < 2:
                    continue
                # 2.2、计算划分后的error值
                now_err = err_cnt(combine_set_1) + err_cnt(combine_set_2)
                # 2.3、更新最优划分
                if now_err < best_err and len(combine_set_1) > 0 and len(combine_set_2) > 0:
                    best_err = now_err
                    bestCriteria = (fea, value)
                    bestSets = (set_1, set_2)

        # 3、判断划分是否结束
        if best_err > self.min_err:
            right = CART_RT(bestSets[0][0],bestSets[0][1], self.min_sample, self.min_err).fit()
            left = CART_RT(bestSets[1][0],bestSets[1][1], self.min_sample, self.min_err).fit()
            return node(fea=bestCriteria[0], value=bestCriteria[1], right=right, left=left)
        else:
            return node(results=leaf(data))  # 返回当前的类别标签作为最终的类别标签

def combine(data_X,data_Y):
    '''样本特征与标签合并
    input:data_X(list):样本特征
          data_Y(list):样本标签
    output:data(list):样本数据
    '''
    m = len(data_X)
    data = copy.deepcopy(data_X)
    for i in range(m):
        data[i].append(data_Y[i])
    return data
        
def err_cnt(data):
    '''回归树的划分指标
    input:  data(list):训练数据
    output: m*s^2(float):总方差
    '''
    data = np.mat(data)
    return np.var(data[:, -1]) * np.shape(data)[0]

def split_tree(data, fea, value):
    '''根据特征fea中的值value将数据集data划分成左右子树
    input:  data(list):训练样本
            fea(float):需要划分的特征index
            value(float):指定的划分的值
    output: (set_1, set_2)(tuple):左右子树的聚合
    '''
    set_1 = []  # 右子树的集合
    set_2 = []  # 左子树的集合
    tmp_11 = []
    tmp_12 = []
    tmp_21 = []
    tmp_22 = []
    for x in data:
        if x[fea] >= value:
            tmp_11.append(x[0:-1])
            tmp_12.append(x[-1])
        else:
            tmp_21.append(x[0:-1])
            tmp_22.append(x[-1])
    set_1.append(tmp_11)
    set_1.append(tmp_12)
    set_2.append(tmp_21)
    set_2.append(tmp_22)
    return (set_1, set_2)

def leaf(dataSet):
    '''计算叶节点的值
    input:  dataSet(list):训练样本
    output: mean(data[:, -1])(float):均值
    '''
    data = np.mat(dataSet)
    return np.mean(data[:, -1])

def predict(sample, tree):
    '''对每一个样本sample进行预测
    input:  sample(list):样本
            tree:训练好的CART回归树模型
    output: results(float):预测值
    '''
    # 1、只是树根
    if tree.results != None:
        return tree.results
    else:
    # 2、有左右子树
        val_sample = sample[tree.fea]  # fea处的值
        branch = None
        # 2.1、选择右子树
        if val_sample >= tree.value:
            branch = tree.right
        # 2.2、选择左子树
        else:
            branch = tree.left
        return predict(sample, branch)

def cal_error(data_X,data_Y, tree):
    ''' 评估CART回归树模型
    input:  data(list):
            tree:训练好的CART回归树模型
    output: err/m(float):均方误差
    '''
    m = len(data_X)  # 样本的个数   
    n = len(data_X[0])  # 样本中特征的个数
    err = 0.0
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(data_X[i][j])
        pre = predict(tmp, tree)  # 对样本计算其预测值
        # 计算残差
        err += (data_Y[i] - pre) * (data_Y[i] - pre)
    return err / m

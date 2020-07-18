import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def euclidien_distence(x, y):
    """计算两个向量x和y的欧氏距离"""
    vec1, vec2 = np.mat(x), np.mat(y)
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    if len(x) == 1:
        return 1
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

def generate_data(dim):
	"""生成dim维的500个数据，数据格式：500*dim"""
	data = [[] for _ in range(500)]
	for i in range(500):
	for j in range(dim):
	    num = random.random()
	    data[i].append(num)
	return data

def main():
	euc_diff_list = [] # 存储欧氏距离下最大最小距离之间的距离
	cos_diff_list = [] # 存储余弦相似度下最大最小相似度之间的距离
	for dim in range(2, 51): # 由于1维情况，无法计算余弦相似度，故从2-50维
	    data = generate_data(dim)
	    euc_distence_list = []
	    cos_distence_list = []
	    for i in range(499):
	        for j in range(i+1, 500):
	            euc_distence_list.append(euclidien_distence(data[i], data[j]))
	            cos_distence_list.append(cosine_similarity(data[i], data[j]))
	    euc_diff_list.append(math.log((max(euc_distence_list) - min(euc_distence_list))/min(euc_distence_list), 10))
	    cos_diff_list.append(math.log((max(cos_distence_list) - min(cos_distence_list))/min(cos_distence_list), 10))

	"""绘图"""
	x = list(range(2, 51))
	plt.plot(x, euc_diff_list, label = '欧氏距离')
	plt.plot(x, cos_diff_list, label = '余弦相似度')
	plt.title('Curse of Dimensionality')
	plt.xlabel('维度')
	plt.ylabel('lg(dif)')
	plt.legend(loc = 'upper right')

if __name__ =="__main__":
	main()
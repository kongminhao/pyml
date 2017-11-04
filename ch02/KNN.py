from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import os


def img_to_vector(filename):
    vector = zeros((1, 1024))
    with open(filename, 'r') as f:
        for i in range(32):  # 一个文件32行
            line_str = f.readline()
            for j in range(32):
                vector[0, 32 * i + j] = int(line_str[j])
    return vector


def hand_write_num_test():
    labels = []
    file_list = os.listdir('digits/trainingDigits')[1:]  # 处理掉mac自带的隐藏文件
    # print(file_list)
    m = len(file_list)
    mat = zeros((m, 1024))# 初始化矩阵
    # os.chdir('digits/trainingDigits')
    for i in range(m):
        filename =  file_list[i]
        num_class = int(filename.split('_')[0])
        labels.append(num_class)
        mat[i, :] = img_to_vector('digits/trainingDigits/%s' % filename)
    # os.chdir("../../")
    test_file_list = os.listdir('digits/testDigits/')[1:]
    # print(test_file_list)
    m_test = len(test_file_list)
    error_count = 0.0
    for i in range(m_test):
        filename = test_file_list[i]
        true_result = int(filename.split('_')[0])
        test_vector = img_to_vector('digits/testDigits/%s' % filename)
        cls_result = classify0(test_vector, mat, labels, 3)
        print("%d %d" % (true_result, cls_result) )
        if true_result != cls_result:
            error_count += 1
    print("总错误数: %d" %error_count)
    print("总错误率: %f" %(error_count/m_test))


def dating_class_test():
    test_ratio = 0.990  # 测试集比率
    data_mat, labels = get_data_set("datingTestSet2.txt")
    norm_mat, ranges, min_val = normalized(data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * test_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        cls_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                               labels[num_test_vecs:m], 3)
        print("%d , %d" % (cls_result, labels[i]))
        if (cls_result != labels[i]):
            error_count += 1
    print("%f" % (error_count / float(num_test_vecs)))


def get_data_set(filename):
    with open(filename, 'r') as f:
        list_line = f.readlines()
        lines = len(list_line)
        mat = zeros((lines, 3))
        labels = []
        index = 0
        for line in list_line:
            # print(line.encode('utf-8'))
            line = line.strip()
            list_form_line = line.split('\t')
            mat[index, :] = list_form_line[0:3]
            labels.append(int(list_form_line[3]))
            index += 1
    return mat, labels


def normalized(dataset):
    min_val = dataset.min(0)
    max_val = dataset.max(0)
    ranges = max_val - min_val
    print(shape(dataset))
    norm_dataset = zeros(shape(dataset))
    m = dataset.shape[0]
    norm_dataset = dataset - tile(min_val, (m, 1))
    norm_dataset = norm_dataset / tile(ranges, (m, 1))
    return norm_dataset, ranges, min_val


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetsize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetsize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    '''
    计算距离
    '''
    sortedDistIndicies = distance.argsort()
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sortedDistIndicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(),
                                key=lambda x: x[1], reverse=True)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    group, labels = createDataSet()
    print(classify0([1, 2], group, labels, 3))
    hand_write_num_test()
    # dating_class_test()
    # group, labels = get_data_set("datingTestSet2.txt")
    # normalized(group)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(group[:,1], group[:,2])
    # plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 17/11/4 下午12:34
from math import log


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels

def calc_shannon_entropy(dataset):
    num_entries = len(dataset)
    label = {}
    for vect in dataset:  # 为所有可能的分类创建字典
        curr = vect[-1]
        if curr not in label.keys():
            label[curr] = 0
        label[curr] += 1
    shannon_entropy = 0.0
    for value in label.values():  # 计算香农熵, 公式不好打，大概就是对所有的值，计算概率，然后对概率求log乘以概率， 求和
        prob = float(value) / num_entries
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


def split_dataset(dataset, axis, value):
    new_dataset = []
    for vec in dataset:
        if vec[axis] == value:
            reduce_vec = vec[:axis]
            reduce_vec.extend(vec[axis + 1:])
            new_dataset.append(reduce_vec)
    return new_dataset


def choose_best_feature(dataset):
    num_feature = len(dataset[0]) - 1  # 最后一位是label
    base_entropy = calc_shannon_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_feature):
        feature_list = [example[i] for example in dataset]
        unique_val = set(feature_list)
        new_entropy = 0.0
        for value in unique_val:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_shannon_entropy(sub_dataset)
        info_gain = base_entropy - new_entropy
        if best_info_gain < info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    class_count = {

    }
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count, key=lambda x: x[1], reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):  # 类别完全相同时停止划分
        return class_list[0]
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    best_feature = choose_best_feature(dataset)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label: {}}
    del (labels[best_feature])
    feature_val = [example[best_feature] for example in dataset]
    unique_val = set(feature_val)
    for value in unique_val:
        sublabels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_dataset(dataset, best_feature, value), sublabels)
    return my_tree

def classify(input_tree, feature_labels, test_vec):
    # print(input_tree.keys())
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feature_index = feature_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feature_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


if __name__ == '__main__':
    dataset , labels = create_dataset()
    label = labels.copy()
    tree = (create_tree(dataset, labels))
    print(tree)
    print(classify(tree, label, [1,0]))
    print(classify(tree, label, [0,0]))

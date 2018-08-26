import numpy as np
import pandas as pd
import re


def dict_cut(text, entities):
    """
    根据词库抽取文本中的实体
    text = '企业提供部门经理的身份证和身份证明'
    entities = ['企业', '部门经理', '身份证', '身份证明']
    return = ['企业', '提', '供', '部门经理', '的', '身份证', '和', '身份证明']
    :param text:文本
    :param entities:实体列表，前面包含后面
    :return:实体抽取结果
    """
    pattern = re.compile('|'.join(entities) + '|.')
    text_entities = pattern.findall(text)

    return text_entities


def dict_locate(text, dictionary={}):
    """
    根据词库做实体定位
    
    text = '企业提供部门经理的身份证和身份证明'
    
    dictionary = {
        '主体': ['企业'],
        '客体': ['部门经理'],
        '材料': ['身份证', '身份证明']
    }
    
    return = [
    {'text': '企业', 'type': '主体', 'location': [2, 3]},
    {'text': '部门经理', 'type': '客体', 'location': [20, 21, 22, 23]},
    {'text': '身份证', 'type': '材料', 'location': [27, 28, 29]},
    {'text': '身份证明', 'type': '材料', 'location': [39, 40, 41, 42]}
    ]
    
    :param text:文本
    :param dictionary:实体库
    :return:实体定位结果
    """
    entities = list(dictionary.values())
    entities_all = []
    entities_len_all = []
    for i in entities:
        for j in i:
            entities_all.append(j)
            entities_len_all.append(len(j))
    entities_index = sorted(range(len(entities_all)),
                            key=lambda x: entities_len_all[x],
                            reverse=True)
    entities_all = np.array(entities_all)[entities_index]
    text_entities = dict_cut(text, entities_all)

    text_location = []
    num = 0
    for text_entity in text_entities:
        for title in dictionary:
            if text_entity in dictionary[title]:
                location = list(range(num, num + len(text_entity)))
                num += len(text_entity)
                text_location.append({'text': text_entity,
                                      'location': location,
                                      'type': title})
                break
    return text_location


def dict_label(text, regulation=[['U', [10]]]):
    """
    根据词库做实体定位
    
    text = '企业提供部门经理的身份证和身份证明'
    
    regulation = [
    ['主体', [1, 2, 3]],
    ['客体', [4, 5, 6]],
    ['材料', [7, 8, 9]],
    ['U', [10]]]
    
    return = [1, 3, 10, 10, 4, 5, 5, 6, 10, 7, 8, 9, 10, 7, 8, 8, 9]
    
    :param text:文本
    :param regulation:标注规则
    :return:实体标注结果
    """
    regulation = {i[0]: i[1] for i in regulation}
    entities = list(dictionary.values())
    entities_all = []
    entities_len_all = []
    for i in entities:
        for j in i:
            entities_all.append(j)
            entities_len_all.append(len(j))
    entities_index = sorted(range(len(entities_all)),
                            key=lambda x: entities_len_all[x],
                            reverse=True)
    entities_all = np.array(entities_all)[entities_index]
    text_entities = dict_cut(text, entities_all)

    text_annotation = []
    for text_entity in text_entities:
        for title in dictionary:
            if text_entity in dictionary[title]:
                if len(text_entity) == 2:
                    text_annotation += [regulation[title][0], regulation[title][2]]
                else:
                    text_annotation += (regulation[title][0:1] +
                                        regulation[title][1:2] * (len(text_entity) - 2) +
                                        regulation[title][2:])
        if text_entity not in entities_all:
            text_annotation += regulation['U']

    return text_annotation


def dict_locate_label(text, dictionary={}, regulation=[['U', [10]]]):
    """
    根据词库做 实体定位+标注

    text = '企业提供部门经理的身份证和身份证明'

    dictionary = {
        '主体': ['企业'],
        '客体': ['部门经理'],
        '材料': ['身份证', '身份证明']
    }

    regulation = [
    ['主体', [1, 2, 3]],
    ['客体', [4, 5, 6]],
    ['材料', [7, 8, 9]],
    ['U', [10]]]

    return = (
    [
    {'location': [0, 1], 'text': '企业', 'type': '主体'}, 
    {'location': [2, 3, 4, 5], 'text': '部门经理', 'type': '客体'}, 
    {'location': [6, 7, 8], 'text': '身份证', 'type': '材料'}, 
    {'location': [9, 10, 11, 12], 'text': '身份证明', 'type': '材料'}
    ], 
    [1, 3, 10, 10, 10, 10, 10, 10, 4, 5, 5, 6, 10, 10, 10, 7, 8, 9, 10, 10, 10, 7, 8, 8, 9])

    :param text:文本
    :param dictionary:实体库
    :param regulation:标注规则
    :return:[实体定位结果, 实体标注结果]
    """
    regulation = {i[0]: i[1] for i in regulation}
    entities = list(dictionary.values())
    entities_all = []
    entities_len_all = []
    for i in entities:
        for j in i:
            entities_all.append(j)
            entities_len_all.append(len(j))
    entities_index = sorted(range(len(entities_all)),
                            key=lambda x: entities_len_all[x],
                            reverse=True)
    entities_all = np.array(entities_all)[entities_index]
    text_entities = dict_cut(text, entities_all)

    text_location = []
    text_annotation = []
    num = 0
    for text_entity in text_entities:
        for title in dictionary:
            if text_entity in dictionary[title]:
                location = list(range(num, num + len(text_entity)))
                num += len(text_entity)
                text_location.append({'text': text_entity,
                                      'location': location,
                                      'type': title})
                if len(text_entity) == 2:
                    text_annotation += [regulation[title][0], regulation[title][2]]
                else:
                    text_annotation += (regulation[title][0:1] +
                                        regulation[title][1:2] * (len(text_entity) - 2) +
                                        regulation[title][2:])
            if text_entity not in entities_all:
                text_annotation += regulation['U']
    return text_location, text_annotation


if __name__ == '__main__':
    text = '企业提供部门经理的身份证和身份证明'
    dictionary = {
        '主体': ['企业'],
        '客体': ['部门经理'],
        '材料': ['身份证', '身份证明']
    }
    entities = ['企业', '部门经理', '身份证明', '身份证']
    regulation = [
        ['主体', [1, 2, 3]],
        ['客体', [4, 5, 6]],
        ['材料', [7, 8, 9]],
        ['U', [10]]]
    print('实体抽取：\n',dict_cut(text, entities))
    print('\n实体定位：\n',dict_locate(text, dictionary))
    print('\n实体标注：\n',dict_label(text, regulation))
    print('\n实体定位+标注：\n',dict_locate_label(text, dictionary, regulation))

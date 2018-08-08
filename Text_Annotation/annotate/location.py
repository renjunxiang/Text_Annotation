import numpy as np
import itertools


def locate(regulations=None, annotation=None):
    """
    标注结果转实体位置
    :param regulations: 标注规则,list,需要符合BMES
        [['n', [1]], ['n', [2, 3, 4]], ['v', [6, 7, 8]]]
    :param annotation: 标注结果,list
        [1, 2, 3, 4, 9, 2, 3, 2, 5, 5, 6, 7, 8, 9]
    :return:定位结果,list
        [[[0], 'n'], [[1, 2, 3], 'n'], [[8], 'v'], [[9, 10, 11], 'v']
    """
    # 规则编码转字典
    regulation_dict = {i[1][0]: i for i in regulations}
    locations = []
    num = 0
    while num < len(annotation):
        entity_start = annotation[num]
        # 找到起始编码
        if entity_start in regulation_dict:
            annotation_type, regulation = regulation_dict[entity_start]
            location = [num]
            num += 1
            # 独立字符
            if len(regulation) == 1:
                locations.append([location, annotation_type])
            else:
                annotation_next = annotation[num]
                # 中间字符
                while annotation_next == regulation[1]:
                    location.append(num)
                    num += 1
                    annotation_next = annotation[num]
                # 是否有终止符，没有就是标注逻辑错误，跳过
                if annotation_next == regulation[-1]:
                    location.append(num)
                    locations.append([location, annotation_type])
                    num += 1
                else:
                    num += 1
                    continue
        else:
            num += 1
            continue

    return locations


def cal_pair_vec(s_vec, l1, l2):
    """
    实体位置转关系特征
    :param s_vec:句子的向量序列,array
    :param l1:实体1位置,list
    :param l2:实体2位置,list
    :return:
    """

    v1 = np.mean(s_vec[l1], axis=0)
    v2 = np.mean(s_vec[l2], axis=0)

    return np.concatenate((v1, v2), axis=0)


def pair_vector(sentence_vector, locations, regular=None):
    """

    :param sentence_vector: 句子向量
        s_vec = np.array([[1, 2, 3, 4],
                          [2, 3, 4, 5],
                          [3, 4, 5, 6],
                          [4, 5, 6, 7],
                          [5, 6, 7, 8],
                          [6, 7, 8, 9]])
    :param locations: 位置信息
        [[[0, 1], 'v'], [[2, 3, 4, 5], 'n'], [[12, 13, 14, 15], 'n']]
    :param regular: 配对规则
        [['v', 'n'], ['n', 'v'], ['v', 'v'], ['n', 'n']]
    :return:
    """
    entity_pairs_raw = itertools.combinations(locations, 2)
    vector_pairs = []
    for entity_pair in entity_pairs_raw:
        if regular:
            # 剔除不符合配对规则的组合
            if [entity_pair[0][1], entity_pair[1][1]] in regular:
                vector_pair = cal_pair_vec(sentence_vector, entity_pair[0][0], entity_pair[1][0])
                vector_pairs.append([vector_pair, [entity_pair[0][1], entity_pair[1][1]]])
        else:
            vector_pair = cal_pair_vec(sentence_vector, entity_pair[0][0], entity_pair[1][0])
            vector_pairs.append([vector_pair, [entity_pair[0][1], entity_pair[1][1]]])

    return vector_pairs


def seq2text(text=None, location=None):
    """

    :param text: 文本
        '从事医疗器械经营管理'
    :param location: 实体位置
        [[[0, 1], 'v'], [[2, 3, 4, 5], 'n'], [[8, 9], 'n']]
    :return: 文本的实体标注
        [从事,v]  [医疗器械,n] 经营 [管理,v]
    """
    result = ''
    for entity_location in location:
        entity = ''
        if entity_location[1] != 'U':
            for character in entity_location[0]:
                entity += text[character]
            entity = ' [%s,%s] ' % (entity, entity_location[1])
        else:
            for character in entity_location[0]:
                entity += text[character]
        result += entity
    return result


if __name__ == '__main__':
    _text = '从事医疗器械经营管理'
    _annotation = [6, 8, 2, 3, 3, 4, 9, 9, 6, 8]
    _regulation = [['n', [1]],
                   ['n', [2, 3, 4]],
                   ['v', [5]],
                   ['v', [6, 7, 8]],
                   ['U', [9]]]
    _s_vec = np.array([[i, i] for i in _annotation])
    _regular = [['v', 'n'], ['n', 'v']]

    print('s_vec:', _s_vec.tolist())
    # [[6, 6], [8, 8], [2, 2], [3, 3], [3, 3], [4, 4], [9, 9], [9, 9], [6, 6], [8, 8]]

    _locations = locate(_regulation, _annotation)
    print('locations:', _locations)
    # [[[0, 1], 'v'], [[2, 3, 4, 5], 'n'], [[12, 13, 14, 15], 'n']]

    _vector_pairs = pair_vector(_s_vec, _locations, _regular)
    print('vector_pairs:', _vector_pairs)
    # [[[7, 7, 3, 3], ['v', 'n']], [[3, 3, 7, 7], ['n', 'v']]]

    annotetion_text = seq2text(_text, _locations)
    print('annotetion_text:', annotetion_text)
    # '[从事,v]  [医疗器械,n] 经营 [管理,n]'

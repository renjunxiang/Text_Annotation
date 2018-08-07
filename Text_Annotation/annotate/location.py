import numpy as np


def locate(regulations=None, annotation=None):
    '''
    标注结果转实体位置
    :param regulations: 标注规则,list,需要符合BMES
    [['n', [1]], ['n', [2, 3, 4]], ['v', [6, 7, 8]]]
    :param annotation: 标注结果,list
    [1, 2, 3, 4, 9, 2, 3, 2, 5, 5, 6, 7, 8, 9]
    :return:定位结果,list
    [[[0], 'n'], [[1, 2, 3], 'n'], [[8], 'v'], [[9, 10, 11], 'v']
    '''
    # 规则编码转字典
    regulation_dict = {i[1][0]: i for i in regulations}
    locations = []
    num = 0
    while num < len(annotation):
        annotation_B = annotation[num]
        # 找到起始编码
        if annotation_B in regulation_dict:
            annotation_type, regulation = regulation_dict[annotation_B]
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


def seq2text(text=None, location=None):
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
    regulation = [['n', [1]],
                  ['n', [2, 3, 4]],
                  ['v', [5]],
                  ['v', [6, 7, 8]],
                  ['U', [9]]]
    annotation = [6, 8, 2, 3, 3, 4, 9, 9, 9, 9, 6, 8]
    location = locate(regulation, annotation)
    print(location)
    # [[[0, 1], 'v'], [[2, 3, 4, 5], 'n'], [[12, 13, 14, 15], 'n']]

    text = '从事医疗器械经营及其管理'
    annotetion_text = seq2text(text, location)
    print(annotetion_text)
    # '[从事,v]  [医疗器械,n] 经营活动及其 [监督管理,n]'

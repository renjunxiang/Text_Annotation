def annotate_merge(p1, p2):
    """
    深度学习的结果作为新词发现,词库匹配结果作为最终修正。
    根据两个实体位置的重合程度判定,完全重合为相同,部分重合采用词库,包含的定义为拓展词,不重合的为新词
    :param p1: 深度学习预测结果
    [{'type': '主体', 'location': [0, 1], 'text': '企业'}]
    :param p2: 词库匹配结果
    [{'type': '客体', 'location': [2, 3, 4, 5], 'text': '部门经理'}]
    :return:
    """
    entity_correct = p2
    entity_new = []
    entity_expand = []
    for p1_i in p1:
        num = 0
        location1 = p1_i['location']
        text1 = p1_i['text']
        bow1 = location1[0]
        eow1 = location1[-1]
        for p2_i in p2:
            text2 = p2_i['text']
            location2 = p2_i['location']
            bow2 = location2[0]
            eow2 = location2[-1]
            if bow1 > eow2:
                continue
            elif bow2 > eow1:
                break
            # 实体位置有重合才比较
            else:
                num += 1
                # 模型预测的词语包含词库的,定义为扩展+词库匹配
                if (text2 in text1) and (len(text2) < len(text1)) \
                        and (text1 not in p2) and (p1_i not in entity_expand):
                    entity_expand.append(p1_i)
        # 没有重合,新实体
        if num == 0:
            entity_new.append(p1_i)
    return entity_new, entity_expand, entity_correct


if __name__ == '__main__':
    _p1 = [
        {'location': [0, 1], 'text': '企业', 'type': '主体'},
        {'location': [2, 3, 4], 'text': '部门经', 'type': '客体'},
        {'location': [6, 7, 8], 'text': '身份证', 'type': '材料'},
        {'location': [9, 10, 11], 'text': '身份证', 'type': '材料'}
    ]
    _p2 = [
        {'location': [0, 1], 'text': '企业', 'type': '主体'},
        {'location': [2, 3, 4, 5], 'text': '部门经理', 'type': '客体'},
        {'location': [9, 10, 11, 12], 'text': '身份证明', 'type': '材料'}
    ]
    _entity_new, _entity_expand, _entity_correct = annotate_merge(_p1, _p2)
    print('\n新词发现：\n')
    for i in _entity_new:
        print(i)
    print('\n拓展词语：\n')
    for i in _entity_expand:
        print(i)
    print('\n词库匹配：\n')
    for i in _entity_correct:
        print(i)

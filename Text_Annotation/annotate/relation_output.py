result = {
    'text': '我喜欢吃苹果',
    'entity': [
        {
            'text': '我',
            'location': [0],
            'type': 'n'
        },
        {
            'text': '喜欢',
            'location': [1, 2],
            'type': 'v'
        },
        {
            'text': '吃',
            'location': [3],
            'type': 'v'
        },
        {
            'text': '苹果',
            'location': [4, 5],
            'type': 'n'
        }
    ],
    'relation': [
        {
            'entity1': {
                'text': '我',
                'location': [0],
                'type': 'n'
            },
            'entity2': {
                'text': '喜欢',
                'location': [1, 2],
                'type': 'v'
            },
            'relation': {
                'score': 0.8,
                'classify': 'n_v'
            }
        },
        {
            'entity1': {
                'text': '吃',
                'location': [3],
                'type': 'v'
            },
            'entity2': {
                'text': '苹果',
                'location': [4, 5],
                'type': 'n'
            },
            'relation': {
                'score': 0.9,
                'classify': 'v_n'
            }
        }
    ]
}

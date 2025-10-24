layout_groups = {
    'bvh8': ['bvh8', 'bvh8-align16', 'bvh8-q8',
             'bvh8-q16', 'bvh8-q8-align16', 'bvh8-q8-ci-align16',
             'bvh8-q8-ci', 'bvh8-q16-align16', 'bvh8-q16-ci-align16'],
    'bvh2': ['sg-eq', 'pbrt', 'pbrt-align16', 'sg-eq-align16',
             'ptr', 'pbrt-soaos', 'pbrt-soaos-align16',
             'pbrt-q16-soaos', 'pbrt-q16'],
    'embree': ['embree-qbvh8i', 'embree-bvh8i'],
}


def retrieve_layout_groups():
    return layout_groups

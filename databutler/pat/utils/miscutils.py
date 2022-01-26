from typing import List, Hashable


def merge_defaultdicts_list(source, other):
    for k, v in other.items():
        source[k].extend(v)


def merge_defaultdicts_dict(source, other):
    for k, v in other.items():
        source[k].update(v)


def merge_defaultdicts_set(source, other):
    for k, v in other.items():
        source[k].update(v)


def remove_duplicates_from_list(elems: List[Hashable]):
    seen = set()
    res = []
    for elem in elems:
        if elem not in seen:
            res.append(elem)
            seen.add(elem)

    return res

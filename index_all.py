import numpy as np
import astropy.units as u
"""
Find the index of all the occurences of an entry in a list
The list can be mad of list of lists or of other array-like
containers (numpy array)
"""


def get_indices(llev):
    aal = []
    for i, a in enumerate(llev[0]):
        for j in range(a):
            aal.append([i, j])
    ilev = 1
    while ilev < len(llev):
        bbl = []
        for i, a in enumerate(llev[ilev]):
            if a == 1:
                bbl.append(aal[i])
            else:
                for j in range(a):
                    lcopy = aal[i].copy()
                    lcopy.append(j)
                    bbl.append(lcopy)
        aal = bbl.copy()
        ilev += 1
    return aal


def flatten_list(l1, return_lev=False):
    """
    >>> import numpy as np
    >>> from index_all import *
    >>> ll = ['a', 'b' , 'v']
    >>> index_all(ll, 'g')
    []
    >>> index_all(ll, 'b')
    1
    >>> ll = ['a', 'b' , 'v', 'a']
    >>> index_all(ll, 'a')
    [0, 3]
    >>> ll = [['a', 'b', ['c', 'd']], ['e', 'f'], 'g']
    >>> index_all(ll, 'd')
    [0, 2, 1]
    >>> ll = [['a', 'b', ['ccc', 'd']], ['e', 'f'], 'gg', 'h']
    >>> index_all(ll, 'gg')
    [2, 0]
    >>> ll = [['a', 'b', ['c', ['d', 'e']]], ['f',
    ...       ['d', 'h']], ['i', 'j'], 'k']
    >>> index_all(ll, 'd')
    [[0, 2, 1, 0], [1, 1, 0]]
    >>> flatten_list(ll)
    ['a', 'b', 'c', 'd', 'e', 'f', 'd', 'h', 'i', 'j', 'k']
    >>> ll = [['a', 3.2, ['c', [4, 'ee']]], ['f',
    ...       [1e-12, 'h']], ['i', 'j'], 'kk']
    >>> index_all(ll, 1e-12)
    [1, 1, 0]
    >>> ll = [2, 3, [5, np.array([7., 8.])], [10, 12]]
    >>> fll, llev = flatten_list(ll, return_lev=True)
    >>> fll
    [2, 3, 5, 7.0, 8.0, 10, 12]
    >>> ind = get_indices(llev)
    >>> index_all(ll, 'g')
    []
    >>> index_all(ll, 'd')
    []
    >>> example = [[[1, 2, 3], 2, [1, 3]], [1, 2, 3]]
    >>> index_all(example, 2)
    [[0, 0, 1], [0, 1], [1, 1]]
    >>> ll = [2, None, [5, np.array([7., 8.])], [10, 12]]
    >>> index_all(ll, None)
    [1, 0]
    """
    islist, llev = [True], []
    while any(islist):
        islist = [type(l2) is list for l2 in l1]
        ll2, len2 = [], []
        for l2 in l1:
            if isinstance(l2, u.Quantity):
                if l2.isscalar:
                    ll2.append(l2)
                    len2.append(1)
                else:
                    ll2.extend(l2)  # extend and len don't work with int
                    len2.append(len(l2))  # or float, and
            elif isinstance(l2, (list, np.ndarray, tuple)):
                ll2.extend(l2)  # extend and len don't work with int
                len2.append(len(l2))  # or float, and
            else:
                ll2.append(l2)
                len2.append(1)
        llev.append(len2)  # with str it is the number of characters!
        l1 = ll2
    if return_lev:
        return l1, llev
    else:
        return l1


def index_all(ll, match, return_list=False):
    """
    Helper function
    """
    if type(ll) is not list:
        return []
    fll, llev = flatten_list(ll, return_lev=True)
    ind = get_indices(llev)
    match_ind = []
    for el, id in zip(fll, ind):
        if el == match:
            if len(llev) == 1:
                match_ind.append(id[0])
            else:
                match_ind.append(id)
    if return_list:
        return match_ind
    if len(match_ind) == 1:
        return match_ind[0]
    else:
        return match_ind


if __name__ == "__main__":
    import doctest
    DOCTEST = True
    doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)
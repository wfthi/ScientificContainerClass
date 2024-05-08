#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
This class contains utility functions for display and broadcasting

Copyright (C) 2023  Wing-Fai Thi (when not specified otherwise)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
# standard library python 3.9
import re
import logging
import inspect
import array
from copy import deepcopy
import pprint
import csv
import warnings
# third-party
import numpy as np
import astropy
from astropy.table import QTable
import pandas as pd
from astropy.io import fits
import astropy.units as u
# local
from index_all import index_all
warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)


def constant(f):
    """
    Function constant that takes an expression, and uses
    it to construct a "getter" - a function that solely
    returns the value of the expression.

    see https://stackoverflow.com/questions/
    2682745/how-do-i-create-a-constant-in-python

    Is is used as a decorator @constant to provide
    a pseudo constant type to Python

    """
    def fset(self, value):
        raise TypeError("Constant cannot be changed")

    def fget(self):
        return f()
    return property(fget, fset)


def to_numpy(d):
    """
    Convert astropy quantities into int, float or numpy arrays
    Keep a list of the units and data types

    Remark
    ------
    VisibleDeprecationWarning: Creating an ndarray from ragged nested
    sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays
    with different lengths or shapes) is deprecated. If you meant to do
    this, you must specify 'dtype=object' when creating the ndarray.

    Parameter
    ---------
    d : dict
        The input dictionary

    Returns
    -------
    dtype : dict
        the keys are the keys of the input dict
        the values are the type of each key

    dunits : dict
        the keys are the keys of the input dict
        the values are the units of each key

    d2 : dict
        the same values as the input dictionary but without
        the units

    Example
    -------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from classUtils import to_numpy
    >>> d = {'a': [1, 2] * u.mm, 'b': 3 * u.deg, 'c': np.array([4, 5])}
    >>> dtype, dunits, obj = to_numpy(d)
    >>> dtype
    {'a': 'float64', 'b': 'float64', 'c': 'int64'}
    >>> dunits
    {'a': Unit("mm"), 'b': Unit("deg"), 'c': ''}
    >>> obj
    {'a': array([1., 2.]), 'b': 3.0, 'c': array([4, 5])}
    >>> d = {'a': [1, 2] * u.mm, 'b': 3 * u.deg,
    ...      'c': np.array([4, 5]), 'd': np.array([])}
    >>> dtype, dunits, obj = to_numpy(d)
    >>> d = {'a': [1, 2] * u.mm, 'b': 3 * u.deg,
    ...      'c': np.array([4, 5]), 'd': ['', 3 * u.m]}
    >>> dtype, dunits, obj = to_numpy(d)
    >>> dunits
    {'a': Unit("mm"), 'b': Unit("deg"), 'c': '', 'd': Unit("m")}
    >>> obj
    {'a': array([1., 2.]), 'b': 3.0, 'c': array([4, 5]), 'd': [None, 3.0]}
    """
    dunits = dict()
    dtype = dict()
    d2 = dict()
    for k, v in d.items():
        dunits[k] = ''
        d2[k] = v
        if isinstance(v, u.Quantity):
            dunits[k] = v.unit
            d2[k] = v.value
        else:
            ind1 = index_all(v, '', return_list=True)
            ind2 = index_all(v, None, return_list=True)
            if ((ind1 != [] and type(ind1[0]) is not list) or
                    (ind2 != [] and type(ind2[0]) is not list)):
                ulist = []
                v3 = []
                for v2 in v:
                    if v2 != '' and v2 is not None:
                        if isinstance(v2, u.Quantity):
                            ulist.append(v2.unit)
                            v3.append(v2.value)
                            dtype[k] = type(v2.value)
                        else:
                            ulist.append('no_units')
                            v3.append(v2)
                            dtype[k] = type(v2)
                    else:
                        v3.append(None)
                uniq_ulist = list(set(ulist))
                if 'no_units' in uniq_ulist:
                    uniq_ulist.remove('no_units')
                if len(uniq_ulist) == 1:
                    dunits[k] = uniq_ulist[0]
                d2[k] = v3

    for k, v in d2.items():
        if isinstance(v, list):
            try:
                v = np.array(v)
            except (ValueError, np.VisibleDeprecationWarning):
                v = np.array(v, dtype=object)
        if isinstance(v, np.ndarray):
            if len(v) > 0:
                vv = v[0]
            else:
                vv = np.array([])
        else:
            vv = v
        # https://stackoverflow.com/questions/
        # 2076343/extract-string-from-between-quotations
        # We assume a numpy array
        if k not in dtype:
            full = re.findall("'([^']*)'", str(type(vv)))[0]
            try:
                dtype[k] = full.split('.')[1]
            except IndexError:
                dtype[k] = full.split('.')[0]
    return dtype, dunits, d2


def len_dict(dict):
    """
    Number of elements in a dictionary

    Parameter
    ---------
    d : dict
        The first dictionary

    Example
    -------
    >>> import numpy as np
    >>> from classUtils import len_dict
    >>> a = {'a' : [2, 3], 'b' : ['c', 'd', 'e']}
    >>> len_dict(a)
    [2, 3]
    >>> b = {'a' : np.array([2, 3]), 'b' : ['c', 'd', 'e']}
    >>> len_dict(b)
    [2, 3]
    """
    lval = []
    s = set()
    for _, value in dict.items():
        if isinstance(value, (str, np.str_)):
            lval.append(1)
        else:
            try:
                lval.append(len(list(value)))
            except TypeError:
                lval.append(1)
        s = set(lval)
    if len(s) == 0:
        return 0
    if len(s) == 1:
        return lval[0]
    else:
        return list(set(lval))


def add_dict(d1, d2, stack=False):
    """
    Routine to add two dictionaries having the same keys.
    It is called by the __add__ method for each class via add_obj.
    The elements of d2 are appended to d1 for each key.

    The main difference between adding and stacking is:

    - adding: similar to extend for a list
    [[1, 2], [3, 4]] + [5, 6] -> [1, 2, 3, 4, 5, 6]

    - stacking: similar to append for a list
    stack([[1, 2], [3, 4]], [5, 6]) -> [[1, 2], [3, 4], [5, 6]]

    Parameter
    ---------
    d1 : dict
        The first dictionary

    d2: dict
        The second dictionary

    stack : bool, optional, default=False
        A special treatment for keys that are arrays: the code stack
        the arrays (of the same number of elements) instead of
        appending them (numpy append)

    Returns
    -------
    dd : dict
        a dictionary withe elements of d2 added at then end of d1

    Notes
    -----
    The routine uses numpy append and works best if the dictionary values
    are numpy arrays

    Examples
    --------
    >>> import sys, io
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from classUtils import add_dict, len_dict
    >>> d1 = {'a': np.array([1, 2]), 'b': 'b'}
    >>> d2 = {'a': 4, 'b': 'v'}
    >>> add_dict(d1, d2)
    {'a': array([1, 2, 4]), 'b': array(['b', 'v'], dtype='<U1')}
    >>> d3 = {'b': 'v', 'a' : 4}
    >>> add_dict(d1, d2)
    {'a': array([1, 2, 4]), 'b': array(['b', 'v'], dtype='<U1')}
    >>> d1 = {'a': np.array([1, 2]), 'b': 'b'}
    >>> d2 = {'a': np.array([1, 4 ,5]), 'b': 'v'}
    >>> add_dict(d1, d2)
    {'a': array([1, 2, 1, 4, 5]), 'b': array(['b', 'v'], dtype='<U1')}
    >>> d1 = {'a': np.array([1, 2]), 'b': 'b'}
    >>> d2 = {'a': np.array([3, 4]), 'b': 'v'}
    >>> d3 = add_dict(d1, d2, stack=True)
    >>> d3['a']
    array([[1, 2],
           [3, 4]], dtype=object)
    >>> d1 = {'a': np.array([1, 2]) * u.mm, 'b': 'b'}
    >>> d2 = {'a': np.array([3, 4]), 'b': 'v'}
    >>> d3 = add_dict(d1, d2) # doctest.ELLIPSIS ...
    >>> # Cannot add quantities and non quantities
    >>> # use stack
    >>> d4 = add_dict(d1, d2, stack=True) # doctest.ELLIPSIS ...
    >>> d1 = {'a': np.array([1, 2]) * u.mm, 'b': 'b'}
    >>> d2 = {'a': np.array([3, 4]) * u.mm, 'b': 'v'}
    >>> d3 = add_dict(d1, d2, stack=True)
    >>> d3['a'][0]
    <Quantity [1., 2.] mm>
    >>> d3['a'][1]
    <Quantity [3., 4.] mm>
    >>> d1 = {'a': np.array([[1, 2], [5, 6]]) * u.mm,
    ...       'b': np.array(['b', 'a'])}
    >>> d2 = {'a': np.array([[3, 4], [8, 9]]) * u.mm,
    ...       'b': np.array(['v', 'w'])}
    >>> d4 = add_dict(d1, d2)
    >>> d4
    {'a': <Quantity [[1., 2.],
               [5., 6.],
               [3., 4.],
               [8., 9.]] mm>, 'b': array(['b', 'a', 'v', 'w'], dtype='<U1')}
    """
    # Check first that the two dictionaries have the same
    # columns
    added, _, _, _ = compare_dict(d1, d2)
    if len(added) > 0:
        mssg = "The two input do not have the same columns"
        logging.warning(mssg)
        logging.warning(f"First input {set(d1.keys())}")
        logging.warning(f"Second input {set(d2.keys())}")
        logging.warning(f"Extra columns {added}")
        logging.warning("Return an empty dictionary")
        return {}

    if stack:
        return stack_dict(d1, d2)
    dd = dict()

    dim1, dim2 = [], []
    for a1, a2 in zip(sorted(d1.items()), sorted(d2.items())):
        b1 = a1[1]
        b2 = a2[1]
        if isinstance(a1[1], list):
            b1 = np.array(b1)
        if isinstance(a2[1], list):
            b2 = np.array(b2)
        try:
            dim1.append(b1.ndim)
        except AttributeError:
            dim1.append(0)
        try:
            dim2.append(b2.ndim)
        except AttributeError:
            dim2.append(0)
    dim1 = np.array(dim1)
    dim2 = np.array(dim2)
    stack1 = dim1 > dim1.min()

    for a1, a2, s1, l1, l2 in zip(sorted(d1.items()),
                                  sorted(d2.items()),
                                  stack1, dim1, dim2):
        b1 = a1[1]
        b2 = a2[1]

        is1 = isinstance(b1, u.Quantity)
        is2 = isinstance(b2, u.Quantity)

        if is1 != is2:
            mssg = "Routine classUtils.add_dict"
            logging.warning(mssg)
            mssg = "Cannot add quantities and non quantities"
            logging.warning(mssg)
            logging.warning(f"First input {b1}")
            logging.warning(f"Second input {b2}")
            logging.warning("Return an empty dictionary")
            logging.warning("Try add with the option stack True")
            return {}

        # Check that the two units are same if the two entries
        # are astropy Quantities
        if is1 and is2:
            if b1.unit != b2.unit:
                logging.warning(f"Columns {a1[0]} have different units")
                logging.warning("Return an empty dictionary")
                logging.warning(f"First units {b1.unit}")
                logging.warning(f"Second units {b1.unit}")
                logging.warning("Try add with the option stack True")
                return {}

        b1 = a1[1]
        b2 = a2[1]
        if (is1 and is2):
            if a1[1].unit != a2[1].unit:  # skip when non-mathcing units
                continue
        if (is1 and is2) or (not is1 and not is2):
            if isinstance(a1[1], list):
                b1 = np.array(b1)
            if isinstance(a2[1], list):
                b2 = np.array(b2)
            if s1:
                if l1 == l2:
                    if l1 > 1:
                        dd[a1[0]] = np.vstack((b1, b2))
                    else:
                        dd[a1[0]] = np.append(b1, b2)
                else:
                    dd[a1[0]] = np.append(b1, b2)
            else:
                dd[a1[0]] = np.append(b1, b2)
        else:
            continue
    return dd


def use_list(dd, only_string=True):
    """
    Use only list instead of numpy arrays, astropy quantiy arrays

    Input
    -----
    dd: dictionary

    only_string: boolean, optional, default=True
        only convert the string numpy arrays into lists

    Returns:
    --------
    : dict
        with the numpy and astropy.Quantities array replaced by lists
        if requested

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from classUtils import use_list
    >>> dd = {'a': np.array([1., 2., 33.]),
    ...       'b': [4., 7., -1.] *u.m,
    ...       'c': np.array(['xx', 'yy', 'zz'])}
    >>> use_list(dd, only_string=False)
    {'a': [1.0, 2.0, 33.0], 'b': [4.0, 7.0, -1.0], 'c': ['xx', 'yy', 'zz']}
    >>> dd = {'a': np.array([1., 2., 33.]),
    ...       'c': np.array(['xx', 'yy', 'zz'])}
    >>> use_list(dd)
    {'a': array([ 1.,  2., 33.]), 'c': ['xx', 'yy', 'zz']}
   """
    d2 = dict()
    for k, v in dd.items():
        if isinstance(v, np.ndarray):
            if isinstance(v[0], str):
                d2[k] = list(v)
                continue
            if only_string:
                d2[k] = v
                continue
            if isinstance(v, astropy.units.quantity.Quantity):
                if not v.isscalar:
                    d2[k] = list(v.value)
                else:
                    d2[k] = v.value
            else:
                d2[k] = list(v)
        else:
            d2[k] = v
    return d2


def stack_dict(d1, d2):
    """
    Stacking instead of adding two dictionaries

    Parameter
    ---------
    d1 : dict
        The first dictionary

    d2: dict
        The second dictionary

    stack : bool, optional
        A special treatment for keys that are arrays: the code stack
        the arrays (of the same number of elements) instead of
        appending them (numpy append)

    Returns:
    --------
    dd : dict
        a dictionary withe elements of d2 added at then end of d1

    Examples
    --------
    >>> import sys, io
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from classUtils import add_dict, len_dict
    >>> d1 = {'a': np.array([1, 2, 3]) * u.mm, 'b': 'b', 'c': 1}
    >>> d2 = {'a': np.array([3, 4]) * u.mm, 'b': 'v', 'c': 2}
    >>> d3 = add_dict(d1, d2, stack=True)
    >>> d4 = add_dict(d2, d1, stack=True)
    >>> d1 = {'a': np.array([[1, 2], [3, 4]]), 'b': 'b', 'c': 1}
    >>> d2 = {'a': np.array([3, 4]), 'b': 'v', 'c': 2}
    >>> d3 = add_dict(d1, d2, stack=True)
    >>> d4 = add_dict(d2, d1, stack=True)
    >>> # consistent size but inconsistent units
    >>> # only add the consistent entry
    >>> d1 = {'a': np.array([[1, 2], [3, 4]])* u.deg, 'b': 'b', 'c': 1}
    >>> d2 = {'a': np.array([3, 4]) * u.mm, 'b': 'v', 'c': 2}
    >>> d3 = add_dict(d1, d2, stack=True)
    >>> # consistent size
    >>> d1 = {'a': np.array([1, 2])* u.mm, 'b': 'a', 'c': 1}
    >>> d2 = {'a': np.array([[3, 4], [5, 8]]) * u.mm,
    ...       'b': ['x', 'v'], 'c': [2, 3]}
    >>> d3 = add_dict(d1, d2, stack=True)
    >>> # inconsistent size
    >>> d1 = {'a': [[1, 2] * u.mm, [3, 4]* u.mm],
    ...       'b': [['b'], ['c']], 'c': [[1], [2]]}
    >>> d2 = {'a': np.array([3, 4, 5]) * u.mm, 'b': 'v', 'c': 2}
    >>> d3 = add_dict(d1, d2, stack=True)
    >>> d4 = add_dict(d2, d1, stack=True)
    >>> #
    >>> d1 = {'a': [[1, 2] * u.mm, [3, 4]* u.mm],
    ...       'b': [['b'], ['c']], 'c': [[1], [2]]}
    >>> d2 = {'a': np.array([[10, 20], [30, 40]]) * u.mm,
    ...       'b': [['v'], ['a', 'c']],
    ...       'c': [[2], [1, 3]]}
    >>> d3 = add_dict(d1, d2, stack=True)
    >>> #
    >>> d1 = {'a': np.array([[1, 2], [3, 4]])* u.mm,
    ...       'b': ['b', 'a'], 'c': [1, 2]}
    >>> d2 = {'a': 1 * u.mm, 'b': 'v', 'c': 2}
    >>> d3 = add_dict(d1, d2, stack=True)
    >>> d1 = {'x' : ['a', 'c'], 'z' :[0, 4]}
    >>> d2 = {'x':['b', 'd'], 'z':[1, 2]}
    >>> d3 = add_dict(d1, d2, stack=True)
    """
    l1keys = list(sorted(d1.keys()))
    l2keys = list(sorted(d2.keys()))

    if l1keys != l2keys:
        logging.warning("Try to combine two dict with different keys")
        logging.warning("Return an empty dictionary")
        return dict()

    _, dunits1, dd1 = to_numpy(d1)
    _, dunits2, dd2 = to_numpy(d2)

    if dunits1 != dunits2:
        dd1 = d1
        dd2 = d2
        to_quantities = False
    else:
        to_quantities = True

    l1 = list(dd1.values())
    len1 = len_dict(dd1)
    try:
        m1 = min(len1)
    except TypeError:
        m1 = 0
    if isinstance(len1, int) and m1 > 1:  # tranpose of a list as a list
        lt1 = list(map(list, zip(*l1)))
    else:
        lt1 = []
        lt1.append(l1)

    l2 = list(dd2.values())
    len2 = len_dict(dd2)
    try:
        m2 = min(len2)
    except TypeError:
        m2 = 0
    if isinstance(len2, int) and m2 > 1:
        lt2 = list(map(list, zip(*l2)))
        for item in lt2:
            lt1.append(item)
    else:
        lt2 = l2
        lt1.append(lt2)

    # transpose of a list
    l3 = list(map(list, zip(*lt1)))
    d3 = dict(zip(l1keys, l3))

    # Try to put back the quantities
    if to_quantities:
        for (k, v), u1 in zip(d3.items(), dunits1.values()):
            try:
                d3[k] = np.array(v, dtype=object)
                if u1 != '':
                    d3[k] *= u.Unit(u1)
            except (np.VisibleDeprecationWarning, ValueError, TypeError):
                if u1 != '':
                    v2 = []
                    for item in v:
                        v2.append(item * u.Unit(u1))
                    try:
                        d3[k] = np.array(v2, dtype=object)
                    except ValueError:
                        d3[k] = v2
    return d3


def compare_dict(d1, d2, order=True):
    """
    Compare two dictionaries

    Parameter
    ---------
    d1 : dict
        The first dictionary

    d2: dict
        The second dictionary

    Results
    -------
    added : set
        a list of extra keys in one of the two dictionaries

    removed : set
        a list of the removed keys

    modified : dict
        a dictionary with the modified values

    same : dict
        a dictionary with the intersect key and values

    Example
    -------
    >>> from classUtils import compare_dict
    >>> a = {'x': [1, 2], 'y': 'ok'}
    >>> b = {'x': [1, 2], 'y': 'ok'}
    >>> c = {'x': [1, 3], 'y': 'ok'}
    >>> d = {'x': [1, 2], 'y': 'not ok'}
    >>> e = {'x': [[1, 2], 3], 'y': ['not ok', 'ok']}
    >>> f = {'x': [[1, 2], [3, 4]], 'y': ['not ok', 'ok']}
    >>> g = {'x': [[1, 2], [3, 4]], 'y': [['not ok'], 'ok']}
    >>> h = {'x': 1, 'y': 'ok'}
    >>> i = {'x': 1, 'y': 'ok', 'z': [1, 2]}
    >>> j = {'x': 1, 'y': 'ok', 'z': [2, 1]}
    >>> added, removed, modified, same = compare_dict(a, b)
    >>> added, removed, modified, same = compare_dict(a, c)
    >>> added, removed, modified, same = compare_dict(a, d)
    >>> added, removed, modified, same = compare_dict(f, g)
    >>> added, removed, modified, same = compare_dict(a, h)
    >>> added, removed, modified, same = compare_dict(h, i)
    >>> added, removed, modified, same = compare_dict(i, h)
    >>> added, removed, modified, same = compare_dict(i, j)

    Notes
    -----
    Modified from
    https://stackoverflow.com/questions/4527942/
    comparing-two-dictionaries-and-checking-how-many-key-value-pairs-are-equal
    to account for values that produce an array of booleans

    It is used to compare two objects

    Warning: the routine finds that ['ok'] and 'ok' are different

    FutureWarning: elementwise comparison failed; returning scalar instead,
    but in the future will perform elementwise comparison. Here the code
    use all() to return a bool.

    The dataclass decorator has been tried but the __eq__ method there does not
    seem tp work with complex keys / values
    """
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = dict()
    same = dict()
    for k in shared_keys:
        try:
            diff = d1[k] == d2[k]
            if not order:
                try:
                    diff = np.sort(d1[k]) == np.sort(d2[k])
                except np.AxisError:
                    diff = d1[k] == d2[k]
        except ValueError:
            diff = False
        if isinstance(diff, (bool, np.bool_)):
            test = diff
        else:
            test = all(np.ravel(diff))  # To deal with the Future warning
        if test:
            same[k] = d1[k]
        else:
            modified[k] = (d1[k], d2[k])
    return added, removed, modified, same


def merge_dict(d1, d2, merge_as_list=False,
               no_units=False,
               same_value_type=False, verbose=False):
    """
    Merge 2 dictionaries made of lists, numpy arrays
    or astropy.Quantities

    One can only merge dictionaries with the same values length
    within each dictionary.
    Trye to use add_dict(d1, d2, stack=True) instead.

    Parameter
    ---------
    d1 : dict
        The first dictionary

    d2: dict
        The second dictionary

    merge_as_list: boolean, optional, default=False
        ignore quantities and numpy array and merge the values
        as a dictionnary of lists

    no_units: boolean, optional, default=False
        return only numpy arrays

    same_value_type: boolean, optional, default=False
        can only merge lists with values of the same type

    verbose: boolean, optional, default=False
        Output error messages if set to True

    Returns
    -------
    : dict
        the merged dictionnary

    Examples
    --------
    >>> import astropy.units as u
    >>> from classUtils import merge_dict
    >>> d1 = {'a': [1, 2, 3], 'b': ['g', 'f', 'e']}
    >>> d2 = {'a': [132, 21, 31], 'b': ['h', 'i', 'j'], 'c': [3, 4, 0]}
    >>> md = merge_dict(d1, d2)
    >>> md['b']
    array(['g', 'f', 'e', 'h', 'i', 'j'], dtype='<U1')
    >>> md = merge_dict(d1, d2, merge_as_list=True)
    >>> md['b']
    ['g', 'f', 'e', 'h', 'i', 'j']
    >>> d1 = {'a': np.array([1, 2, 3]), 'b': ['g', 'f', 'e']}
    >>> d2 = {'a': np.array([132, 21, 31]), 'b': ['h', 'i', 'j'],
    ...       'c': [3, 4, 0]}
    >>> md = merge_dict(d1, d2)
    >>> md['a']
    array([  1,   2,   3, 132,  21,  31])
    >>> md = merge_dict(d1, d2, merge_as_list=True)
    >>> md['a']
    [1, 2, 3, 132, 21, 31]
    >>> d1 = {'a': [1, 2, 3] * u.cm, 'b': ['g', 'f', 'e']}
    >>> d2 = {'a': [132, 21, 31] * u.cm, 'b': ['h', 'i', 'j'], 'c': [3, 4, 0]}
    >>> md = merge_dict(d1, d2)
    >>> md['a']
    <Quantity [  1.,   2.,   3., 132.,  21.,  31.] cm>
    >>> md = merge_dict(d1, d2, no_units=True)
    >>> md['a']
    array([  1.,   2.,   3., 132.,  21.,  31.])
    >>> d1 = {'a': [1, 2, 3], 'b': ['g', 'f', 'e']}
    >>> d2 = {'a': [132, 21, 31] * u.cm, 'b': ['h', 'i', 'j'], 'c': [3, 4, 0]}
    >>> md = merge_dict(d1, d2, merge_as_list=True)
    >>> md['a']
    [1, 2, 3, 132.0, 21.0, 31.0]
    >>> d1 = {'b': [1, 2, 3], 'a': ['g', 'f', 'e']}
    >>> d2 = {'a': [132, 21, 31], 'b': ['h', 'i', 'j'], 'c': [3, 4, 0]}
    >>> md = merge_dict(d1, d2)
    >>> md['a']
    array(['g', 'f', 'e', '132', '21', '31'], dtype='<U21')
    >>> md['a']
    array(['g', 'f', 'e', '132', '21', '31'], dtype='<U21')
    >>> md = merge_dict(d1, d2, merge_as_list=True)
    >>> md['a']
    ['g', 'f', 'e', 132, 21, 31]
    >>> md = merge_dict(d1, d2, same_value_type=True, verbose=True)
    Dictionary with different types of values
    Set same_value_type=False to ignore the check
    >>> d1 = {'a': [1, 2, 3] * u.cm, 'b': ['g', 'f', 'e']}
    >>> d2 = {'a': [132, 21, 31] * u.cm, 'b': ['h', 'i', 'j'],
    ...       'c': [3, 4, 0] * u.K}
    >>> md = merge_dict(d1, d2)
    >>> d1 = {'a': [1, 2, 3] * u.cm, 'b': ['g', 'f', 'e']}
    >>> d2 = {'a': [132, 21, 31] * u.cm, 'b': ['h', 'i', 'j'],
    ...       'c': [3, 4, 0]}
    >>> md = merge_dict(d1, d2)
    """
    keys = list(d1.keys())
    keys.extend(list(d2.keys()))
    keys = list(set(keys))

    dtype1, dunits1, _ = to_numpy(d1)
    dtype2, dunits2, _ = to_numpy(d2)

    # merge only if the values are of the same type
    if same_value_type:
        same_type = {k: True for k in keys}
        for k in keys:
            if k in dtype1:
                if k in dtype2:
                    if dtype1 == dtype2:
                        same_type[k] = True
                    else:
                        same_type[k] = False
        if not all(list(same_type.values())):
            print('Dictionary with different types of values')
            print('Set same_value_type=False to ignore the check')
            return None

    # merge only the values that have the same units
    if not no_units and not merge_as_list:
        same_units = {'k': True for k in keys}
        for k in keys:
            if k in dunits1:
                if k in dunits2:
                    if dunits1[k] == dunits2[k]:
                        same_units[k] = True
                    else:
                        same_units[k] = False
        if not all(list(same_units.values())):
            if verbose:
                logging.error('Values do not have the same input units')
                logging.error('Set no_units=True or merge_as_list=True\
                      to ignore the units')
            return None

    # Check that the length of the values are the same for all the keys
    ld1, ld2 = [], []
    for k1, l1 in d1.items():
        try:
            ld1.append(len(l1))
        except TypeError:
            d1[k1] = [d1[k1]]
            ld1.append(1)
    if len(set(ld1)) == 1:
        ld1 = ld1[0]
    else:
        print('Values of different lengths between lists in the dict 1')
        return None
    for k2, l2 in d2.items():
        try:
            ld2.append(len(l2))
        except TypeError:
            d2[k2] = [d2[k2]]
            ld2.append(1)
    if len(set(ld2)) == 1:
        ld2 = ld2[0]
    else:
        print('Values of different lengths between lists in the dict 2')
        return None

    dunits = {}
    new_dict = {}
    for k in keys:
        if k in dunits1:
            dunits[k] = dunits1[k]
        elif k in dunits2:
            dunits[k] = dunits2[k]
        else:
            print("Something wrong happened")
            return None
        if k in d1:
            if dunits1[k] != '':
                new_dict[k] = list(d1[k].value).copy()
            else:
                new_dict[k] = list(d1[k]).copy()
        else:
            new_dict[k] = [None] * ld1
        if k in d2:
            if dunits2[k] != '':
                new_dict[k].extend(list(d2[k].value))
            else:
                new_dict[k].extend(list(d2[k]))
        else:
            new_dict[k].extend([None] * ld2)
        if not merge_as_list:
            new_dict[k] = np.array(new_dict[k])
            if not no_units:
                if dunits[k] != '':
                    if None in new_dict[k]:
                        val = []
                        for v in new_dict[k]:
                            if v is not None:
                                val.append(v * u.Unit(dunits[k]))
                            else:
                                val.append(None)
                        new_dict[k] = val
                    else:
                        new_dict[k] *= u.Unit(dunits[k])
    return new_dict


class classUtils:
    """
    Base class with special methods. Basically classUtils
    extends the dictionary class

    It contains some basic array manipulation methods,
    conversion to other popular data containers (astropy Table,
    panda Dataframe), and input/output from files of various formats
    (cvs, fits).

    The class is an iterator.

    It is used as parent class. A simple way to use it is

    >>> from classUtils import classUtils
    >>> class MyClass(classUtils):
    ...     def __init__(self, **kwargs):
    ...         super().__init__(**kwargs)

    Create a Car class based on classUtils
    >>> from classUtils import classUtils
    >>> class Car(classUtils):
    ...     def __init__(self, **kwargs):
    ...         super().__init__(**kwargs)
    >>> dealer1_dict = {'Mercedes': 2, 'Toyota': 4, 'BMW': 0}
    >>> dealer2_dict = {'Mercedes': 1, 'Toyota': 2, 'BMW': 1}
    >>> dealer1 = Car(**dealer1_dict)
    >>> dealer1.name = 'Jane'
    >>> dealer2 = Car(**dealer2_dict)
    >>> dealer2.name = 'John'
    >>> dealers = dealer1 + dealer2
    >>> # Transform into a Pandas DataFrame
    >>> df_dealers = dealers.to_pandas()

    * check_consistency
    * copy
    * to_dict
    * keys
    * argsort
    * sort
    * arglexsrot
    * lexsort
    * attribute_type
    * to_table
    * pprint
    * to_pandas
    * drop
    * compare
    * type
    * number_columns = number_attributes
    * select_columns
    * remove column
    * from_table
    * save
    * from_fits : call to from_file
    * from_file
    * from_pandas
    * rename_column
    * to_hdu
    * doc
    * operation overload = +, ==, !=, len
    * stack
    * to_numpy
    * to_csv
    * from_csv
    * object_to_str
    * merge
    """
    def __init__(self, **kwargs):
        """
        Instantiate a class from a dictionary with keys compatible with the
        class. Extra keys can be present but the essential kets should be in
        the dictionary for the created object to access the class methods.

        Parameters
        ----------
        **kwargs : dict
            a dictionary to be converted to an object

        Notes
        -----
        The class __init__ should be such that no explicit parameters are need
        to create an instance of the class, i.e. that there is no explicit
        arguments.
        Keyword arguments are possible, for example
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 1], y=2, z=[0 ,4, 5])
        >>> obj2 = Test(x=['a', 0], y=2, z=[0 ,4, 5])
        >>> d1 = obj1.__dict__
        >>> obj = Test(**d1)
        >>> obj
        {'x': ['a', 1], 'y': 2, 'z': [0, 4, 5]}
        """
        self.__dict__.update(kwargs)

    def __repr__(self):
        """
        Display the object as a dictionary

        Examples
        --------
        >>> import astropy.units as u
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'x': [['a', 1], ['b', 2]], 'y': [0, 4]}
        >>> obj1 = Test(**d)
        >>> print(obj1)
        {'x': [['a', 1], ['b', 2]], 'y': [0, 4]}
        """
        return str(self.__dict__)

    def _set_one_(self, key, value):
        """
        Allow attribution of the type:
            v2[0:2] = vis[200:202]
        like in Numpy but for objects of classes inheritating from classUtils

        Parameters
        ----------
        key : any valid type (int, slide, array, numpy array)
            t[key] = value

        value : various type

        Examples
        --------
        >>> import astropy.units as u
        >>> import numpy as np
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'x': np.array([1, 2, 3, 4, 5]),
        ...      'y': np.array(['a', 'b', 'c', 'd', 'e'])}
        >>> e = {'x': np.array([11, 12, 13, 14, 15]),
        ...      'y': np.array(['f', 'g', 'h', 'i'])}
        >>> obj1 = Test(**d)
        >>> obj2 = Test(**e)
        >>> obj3 = obj1.copy()
        >>> obj1[0:3] = obj2[1:4]
        >>> obj1['x']
        array([12, 13, 14,  4,  5])
        >>> obj1['y']
        array(['g', 'h', 'i', 'd', 'e'], dtype='<U1')
        >>> d = {'x': np.array([1, 2, 3, 4, 5]),
        ...      'y': np.array(['a', 'b', 'c', 'd', 'e'])}
        >>> obj1 = Test(**d)
        >>> w = np.array([True, False,  False,  True,  True])
        >>> obj1[w] = obj2[[1, 2, 3]]
        >>> obj1['x']
        array([12,  2,  3, 13, 14])
        >>> obj1['y']
        array(['g', 'b', 'c', 'h', 'i'], dtype='<U1')
        >>> obj1['z'] = np.array([100, 101, 102, 103, 104, 105])
        >>> obj1.z
        array([100, 101, 102, 103, 104, 105])
        >>> w = np.array([True, True, False, False, False])
        >>> obj3[w] = obj2[0:2]
        >>> obj3[0]
        {'x': 11, 'y': 'f'}
        """
        a = dict()
        for k, v in self.to_dict().items():
            try:
                v[key] = value[k]  # replace by value[k]
            except TypeError:
                continue
            a[k] = v
        self.__dict__.update(a)

    def _set_indices_(self, key, value):
        """
        Get item for indices in a list or numpy array
        """
        if isinstance(key[0], (bool, np.bool_)):
            i = np.arange(0, len(key), 1)
            j = i[key]
        else:
            j = key
        return self._set_one_(j, value)

    def __setitem__(self, key, value):
        """
        It allows to access data like t['key_name'] = value
        or update an object using t[1] = v[2], t[0:2] = v[10:12]

        Examples
        --------
        see _set_one_(self, key, value):
        """
        if isinstance(key, str):
            setattr(self, key, value)  # to allow t['key_name'] = 1
        elif isinstance(key, (int, np.int32, np.int64, slice)):
            self._set_one_(key, value)  # t[1] = v[2]
        elif isinstance(key, (list, np.ndarray, tuple, array)):
            self._set_indices_(key, value)  # t[0:2] = v[10:12]
        else:
            raise KeyError('Unknown key/indice type:', type(key))

    def _getitem_one(self, item):
        """
        Get item from int/mask item
        """
        a = dict()
        if isinstance(item, tuple):
            item = list(item)
        for k, v in self.to_dict().items():
            try:
                vtype = type(v)
                if vtype == list:
                    v = np.array(v)
                    vv = v[item]
                    if isinstance(vv, np.ndarray):
                        a[k] = list(vv)
                    else:  # single value output
                        a[k] = vv
                else:  # numpy return a float/str/int
                    a[k] = v[item]  # instead of a single entry array
            except TypeError:
                continue
        try:
            v = self.__class__()
        except TypeError:
            v = self.copy()   # create a new object
        v.__dict__.update(a)  # update the object via a dictionary
        return v

    def _getitem_indices(self, item):
        """
        Get item for indices in a list or numpy array

        Examples
        --------
        >>> import numpy as np
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'x': np.array([1, 2, 3, 4, 5]),
        ...      'y': np.array(['a', 'b', 'c', 'd', 'e'])}
        >>> e = {'x': [11, 12, 13, 14, 15],
        ...      'y': ['f', 'g', 'h', 'i']}
        >>> obj1 = Test.from_dict(d)
        >>> obj2 = Test(**e)  # alternative to from_dict()
        >>> obj1[1]
        {'x': 2, 'y': 'b'}
        >>> obj1[1:3]
        {'x': array([2, 3]), 'y': array(['b', 'c'], dtype='<U1')}
        >>> obj1[[1, 3, 0]]
        {'x': array([2, 4, 1]), 'y': array(['b', 'd', 'a'], dtype='<U1')}
        >>> obj2[[0, 3]]  # list of indices
        {'x': [11, 14], 'y': ['f', 'i']}
        >>> obj2[0:3]  # slice
        {'x': [11, 12, 13], 'y': ['f', 'g', 'h']}
        >>> obj2[2]  # index
        {'x': 13, 'y': 'h'}
        """
        if isinstance(item[0], (bool, np.bool_)):
            i = np.arange(0, len(item), 1)
            j = i[item]
        else:
            j = item
        return self._getitem_one(j)

    def __getitem__(self, item):
        """
        Allow indexing and slicing of the object

        Examples
        --------
        >>> import numpy as np
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'x': np.array([1, 2, 3, 4, 5]),
        ...      'y': np.array(['a', 'b', 'c', 'd', 'e'])}
        >>> e = {'x': [11, 12, 13, 14, 15],
        ...      'y': ['f', 'g', 'h', 'i']}
        >>> obj1 = Test.from_dict(d)
        >>> obj2 = Test(**e)  # alternative to from_dict()
        >>> obj1.keys()
        ['x', 'y']
        >>> obj1['x']
        array([1, 2, 3, 4, 5])
        >>> obj1[[0, 2, 1]]
        {'x': array([1, 3, 2]), 'y': array(['a', 'c', 'b'], dtype='<U1')}
        >>> obj2[(0, 2)]
        {'x': [11, 13], 'y': ['f', 'h']}
        """
        if isinstance(item, str):  # allow acces via t['key_name']
            #  equivalent to OrderedDict.__getitem__(self.to_dict(), i)
            return getattr(self, item)

        elif isinstance(item, (int, np.int32, np.int64, slice)):
            return self._getitem_one(item)

        elif isinstance(item, (list, np.ndarray, tuple, array)):
            return self._getitem_indices(item)

        else:
            raise KeyError('Unknown item type:', type(item))

    def clean_quantities(self):
        """
        Private method

        transform list of the same quantities into a quantity array
        """
        ulist = []
        for k1, v1 in self.items():
            if isinstance(v1, list):
                for v2 in v1:
                    if isinstance(v2, astropy.units.quantity.Quantity):
                        ulist.append(v2.unit)
                    else:
                        ulist.append('')
                uniq_ulist = list(set(ulist))
                if '' not in ulist and len(list(set(ulist))) == 1:
                    self[k1] = [v2.value for v2 in v1] * u.Unit(uniq_ulist[0])
        return self

    def __len__(self):
        """
        Compute the length (ie the number of rows).

        Notes
        -----
        It calls _nelements(). It does not work when
        the different values are arrays of different lengths
        One needs to use ._nelements() instead

        Examples
        --------
        >>> import numpy as np
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'x': np.array([1, 2, 3, 4, 5]),
        ...      'y': np.array(['a', 'b', 'c', 'd', 'e'])}
        >>> e = {'x': [11, 12, 13, 14, 15],
        ...      'y': ['f', 'g', 'h', 'i']}
        >>> f = {'x': np.array([11, 12, 13, 14, 15]),
        ...      'y': np.array(['f', 'g', 'h', 'i'])}
        >>> obj1 = Test.from_dict(d)
        >>> obj2 = Test(**e)  # alternative to from_dict()
        >>> obj3 = Test(**f)
        >>> len(obj1)
        5
        >>> print(obj1._nelements())
        5
        >>> print(obj2.__len__())
        [5, 4]
        >>> print(obj2._nelements())
        [5, 4]
        """
        return self._nelements()

    def _nelements(self, exclude=[]):
        """
        Calculate the number of entries (rows) of an object. If the columns
        are of different lengths, list the set of the lengths

        Examples
        --------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, x, y, z):
        ...         self.x = x
        ...         self.y = y
        ...         self.z = z
        >>> obj = Test(['a', 1], 2, [0 ,4, 5])
        >>> print(obj._nelements())
        [2, 1, 3]
        >>> obj._nelements(exclude=['y'])
        [2, 3]
        >>> obj._nelements(exclude=['x', 'y', 'z'])
        0
        >>> obj = Test(['a', 1], 2, [0 ,4])
        >>> print(obj._nelements())
        [2, 1, 2]
        >>> obj = Test(['a', 1], [2, 3], [0 ,4])
        >>> print(obj._nelements())
        2
        """
        lval = []
        for key, value in self.__dict__.items():
            if (key not in exclude):
                if isinstance(value, (str, np.str_)):
                    lval.append(1)
                else:
                    try:
                        lval.append(len(list(value)))
                    except TypeError:
                        lval.append(1)
        if len(lval) == 0:
            return 0
        if len(lval) == 1:
            return lval[0]
        else:
            if len(set(lval)) == 1:
                return lval[0]
            else:
                return list(lval)

    def __add__(self, other, stack=False):
        """
        Addition/Stacking/Merging of two objects
        It is an operator overwriting way to perform the task

        Notes
        -----
        The attributes are Numpy objects

        Parameters
        ----------
        other : classUtils class
            the object to be added to the first object

        stack : bool, optional, default=False
            stack instead of numpy appending

        Returns
        -------
        obj : classUtils class
            append another object to self

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0 ,4])
        >>> obj2 = Test(x=['b', 'd'], z=[1 ,2])
        >>> obj3 = obj1 + obj2
        >>> obj4 = obj1.__add__(obj2)
        >>> obj3 == obj4
        True
        >>> obj5 = obj1.__add__(obj2, stack=True)
        """
        added, removed, _, _, _ = self.compare(other)
        if list(added) == [] and list(removed) == []:
            obj = self.copy()
            obj.__dict__.update(add_dict(self.__dict__, other.__dict__,
                                         stack=stack))
            return obj
        else:
            logging.warning("Two objects have different number of columns")
            # return a list
            return [self, other]

    def __iadd__(self, other):
        """
        Addition with assignment

        Compared to obj1 = obj1 + obj2, it avoid the creation of a new
        object

        Parameters
        ----------
        other : visit class
            the object to be added to the first object

        Returns
        -------
        self : classUtils class
            append another object to self and return itself + other

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0 ,4])
        >>> obj2 = Test(x=['b', 'd'], z=[1 ,2])
        >>> obj1 += obj2  # <-> obj1 = obj1 + obj2
        >>> all(obj1['x'] == np.array(['a', 'c', 'b', 'd']))
        True
        """
        self.__dict__.update(add_dict(self.__dict__, other.__dict__))
        return self

    def check_column_type(self, verbose=False):
        """
        Check that the columns are numpy arrays or astropy quantites
        for objects with more than one row

        This is useful if one wants to save the object as a fits file.

        Parameter
        ---------
        self : object of class classUtils or inheritating from classUtils
            the current object

        verbose : bool, optional, default=False
            if True, a list of the keys and the column lenghts will be printed
            together if the column conforms to the object format

        Example
        -------
        >>> import astropy.units as u
        >>> from astropy.time import Time
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=0)
        >>> obj1.check_column_type()
        True
        >>> obj2 = Test(x=['a', 'c'], z='c')
        >>> obj2.check_column_type()
        True
        >>> obj3 = Test(x=[1, 2] * u.mm, z='c')
        >>> obj3.check_column_type()
        True
        >>> obj4 = Test(x=[1, 2] * u.mm, z=np.array([3, 4, 5]))
        >>> obj4.check_column_type()
        True
        >>> obj5 = Test(x=[1, 2] * u.mm, z=np.array([3, 4, 5]))
        >>> obj5.check_column_type()
        True
        >>> t = Time([2022.2, 2022.3], format='decimalyear')
        >>> obj6 = Test(x=[1, 2] * u.mm, z=t)
        >>> obj6.check_column_type(verbose=False)
        False
        """
        n = self._nelements()
        # there are columns with more than one value
        ok_global = True
        if isinstance(n, list):
            nb = n[0]
        else:
            nb = n
        if nb > 0:
            d = self.__dict__
            for k, v in d.items():
                if isinstance(v, str):
                    ok = True
                else:
                    try:
                        lv = len(v)
                    except TypeError:
                        lv = 0
                    if lv > 0:
                        ok = isinstance(v, (list, np.ndarray,
                                            astropy.units.quantity.Quantity))
                        if ok is False:
                            ok_global = False
                    else:
                        ok = True
                if verbose:
                    print(k, ' - ', type(v), ' : ', ok)
        else:
            if verbose:
                print('Only one row')
            ok_global = True

        return ok_global

    def check_consistency(self, remove_inconsistency=False,
                          keep_one=False, verbose=False):
        """
        Check that all the attributes have the same number of values
        Optionally remove the attributes that are single value]

        Parameters
        ----------
        remove_inconsistency : bool, optional, default=False
            inplace removal of all non array attributes

        keep_one : bool, optional, default=False
            inplace removal of all array attributes. This has the
            opposite effect than remove_inconsistency

        verbose : bool, optional, default=False
            if True, a list of the keys and the column lenghts will be printed

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=0)
        >>> obj1.check_consistency(verbose=False)
        False
        >>> obj1 = Test(x=['a', 'c'], y=[2, 3], z=0)
        >>> ok = obj1.check_consistency(remove_inconsistency=True)
        >>> obj1
        {'x': ['a', 'c'], 'y': [2, 3]}
        >>> obj1 = Test(x=[2, 1, 5, 6, 5, 3], y=[2, 5, 6, 8, 0, 9],
        ...             z=[0, 4, 5, 9, 1])
        >>> ok = obj1.check_consistency()
        """
        lv = []
        n = self._nelements()
        copy = self.copy()
        if isinstance(n, list):
            for key in copy.__dict__.keys():
                try:
                    lvalue = len(copy[key])
                except TypeError:
                    lvalue = 1
                if isinstance(copy[key], (str, np.str_)):
                    lvalue = 1  # do not count the number of char in strings
                lv.append(lvalue)
                if (remove_inconsistency):
                    if lvalue == 1:
                        delattr(self, key)
                if (keep_one):
                    if lvalue > 1:
                        delattr(self, key)
                if verbose:
                    print(key, ' :', lvalue)
            return len(set(lv)) == 1
        else:
            return True

    def copy(self):
        """
        Make a copy of the object (Do not use the = operator!)
        Note it uses the intrinsic deepcopy function

        Parameters
        ----------
        self : the entire object

        Returns
        -------
        obj : a deep copy of the input object

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=0)
        >>> obj2 = obj1.copy()
        >>> obj2 == obj1
        True
        """
        return deepcopy(self)

    def to_dict(self):
        """
        Convert the object into a dictionary
        self.__dict__ is equivalent to vars(self)

        Parameters
        ----------
        self : an instance of the class

        Returns
        -------
        dict : a dictionary
        """
        # https://stackoverflow.com/questions/1251692/
        # how-to-enumerate-an-objects-properties-in-python
        return self.__dict__

    def keys(self):
        """
        Return the keys of the object

        Parameters
        ----------
        self : an instance of the class

        Returns
        -------
        : list of keys
        """
        return list(self.__dict__.keys())

    def values(self):
        """
        Return the keys of the object

        Parameters
        ----------
        self : an instance of the class

        Returns
        -------
        : list of values
        """
        return list(self.__dict__.values())

    def items(self):
        """
        Return the keys of the object

        Parameters
        ----------
        self : an instance of the class

        Returns
        -------
        : list of items
        """
        return list(self.__dict__.items())

    def keys_to_screen(self):
        """
        Show the keys of the object

        Parameters
        ----------
        self : an instance of the class

        Returns
        -------
        screen output
        """
        names = (f"'{x}'" for x in self.__dict__.keys())
        return "<{1} names=({0})>".format(",".join(names),
                                          self.__class__.__name__)

    def argsort(self, key):
        """
        Compute the index for the sort with values at key

        Parameters
        ----------
        key : str
            the key used to sort the class rows

        Returns
        -------
        Numpy array of int
            indices to sort an array

        Example
        -------
        >>> from classUtils import classUtils
        >>> class MyClass(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d1 = {'a' : [1, 2], 'b' : ['qq', 'tt']}
        >>> data1 = MyClass.from_dict(d1)
        >>> data1.argsort('a')
        array([0, 1])
        """
        if key not in self.keys():
            logging.warning(f"Key '{key}' not found. Return None")
            return None
        if not isinstance(len(self), int):
            logging.warning("Unconsistent size. No sorting")
            return None
        return np.argsort(self[key])

    def sort(self, key, inplace=True):
        """
        Sort the object according to the key value

        Parameters
        ----------
        key : str
            the key used to sort the class object rows

        inplace : bool, optional, default=True
            if True the input object is updated inplace

        Returns
        -------
        self : class object
            if inplace=True, the sorted obj

        obj : class object
            if inplace=False, the sorted obj is provided as a new class
            object

        Notes
        -----
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c', 'b'], z=[0 ,4, 1])
        >>> obj1.argsort('z')
        array([0, 2, 1])
        >>> obj1.sort('z')
        >>> obj1
        {'x': ['a', 'b', 'c'], 'z': [0, 1, 4]}
        """
        self_dict = self.to_dict()
        if key not in self_dict:
            logging.warning(f"Key '{key}' not found. No sorting")
            return self
        if not isinstance(len(self), int):
            logging.warning("Unconsistent size. No sorting")
            return self
        obj = self[self.argsort(key)]
        if inplace:
            self.__dict__.update(obj.to_dict())
        else:
            return obj

    def arglexsort(self, keys):
        """
        Compute the index for the lexsort with keys

        Parameters
        ----------
        self : class object

        keys : tuple or list of str
            the keys used to sort the class rows
            the right most key is the first to be used
            then the other keys starting from the right will
            be used

        Returns
        -------
        indices(N,) : ndarray of ints
            Array of indices that sort the keys along the specified axis.

        Notes
        -----
        see Numpy lexsort for more explanations
        """
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if key not in self.to_dict():
                logging.warning(f"Key '{key}' not found. Return None")
                return None
        if not isinstance(len(self), int):
            logging.warning("Unconsistent size. No sorting")
            return None
        return np.lexsort([self[k] for k in keys])

    def lexsort(self, keys, inplace=True):
        """
        Sort the object according to the key values
        starting from the right

        It is similar to the function groupy in the
        Pandas package

        Parameters
        ----------
        keys : str or a list/tuple of keys
            the keys to be used to sort the class object rows

        Returns
        -------
        self : class object
            if inplace=True, the sorted obj

        obj : class object
            if inplace=False, the sorted obj is provided as a new class
            object

        inplace : bool, optional, default=True
            if True the input object is updated inplace

        Example
        -------
        >>> import numpy as np
        >>> import astropy.units as u
        >>> from classUtils import classUtils
        >>> class Ctest3(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> test = Ctest3(x=np.array([5, 1, 2, 3, 3, 4]),
        ...               y=np.array(['e', 'a', 'b', 'e', 'c', 'd']),
        ...               z=[15, 11, 12, 13, 15, 14] * u.m)
        >>> out = test.lexsort(['y', 'x'], inplace=False)
        >>> all(out['x'] == np.array([1, 2, 3, 3, 4, 5]))
        True
        >>> all(out['y'] == np.array(['a', 'b', 'c', 'e', 'd', 'e']))
        True
        """
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if key not in self.to_dict():
                logging.warning(f"Key '{key}' not found. No sorting")
                return self
        if not isinstance(len(self), int):
            logging.warning("Unconsistent size. No sorting")
            return self
        obj = self[self.arglexsort(keys)]
        if inplace:
            self.__dict__.update(obj.to_dict())
        else:
            return obj

    def number_columns(self):
        return self.number_attributes()

    def number_attributes(self):
        """
        Give the number of columns (attributes)
        """
        return len(self.__dict__.keys())

    def info(self):
        """
        Synonym for attribute_type()
        """
        return self.attribute_type()

    def attribute_type(self):
        """
        List the type of the attributes

        Returns
        -------
        d : dict of keys and the type of the values

        Notes
        -----
        The method returns a dictionary instead of a print
        """
        d = dict()
        for key in self.__dict__.keys():
            if isinstance(self[key], astropy.units.quantity.Quantity):
                units = self[key].unit.to_string()
                d[key] = 'astropy.quantity : ' + units
            else:
                d[key] = type(self[key])
        return d

    def to_table(self, remove_inconsistency=False):
        """
        Transform a class object into an astropy Table

        Astropy Table objects can handle astropy object with
        units

        Parameters
        ----------
        remove_inconsistency : bool, optional, default=False
            Remove columns with length different than the rest.
            Only keep the columns with the full length
            Without this the code will fail to make a table

        returns
        -------
        QTable : `astropy.table.table.QTable`
            Astropy Quantity Table

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0 ,4])
        >>> tab = obj1.to_table()
        >>> all(tab['x'] == ['a', 'c'])
        True
        >>> all(tab['z'] == [0, 4])
        True
        >>> obj1 = Test(x=['a', 'c'], y=[2, 3] * u.deg, z=0)
        >>> tab = obj1.to_table(remove_inconsistency=True)
        >>> tab['x'][0]
        'a'
        >>> tab['x'][1]
        'c'
        """
        copy = self.copy()
        ok = copy.check_consistency(remove_inconsistency=remove_inconsistency)
        if (not ok) and (remove_inconsistency is False):
            print("Warning: Inconsistent table")
            print("You can use object.to_table(remove_inconsistency=True)")
            return QTable()
        return QTable(copy.to_dict())

    def object_to_str(self):
        """
        Convert numpy array of object type to the string type

        Parameter
        ---------
        self : classUtils object

        Example
        -------
        >>> import numpy as np
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=np.array(['a', 'c ']).astype('object'), z=[0, 4])
        >>> obj1.object_to_str().save('test.fits', overwrite=True)
        >>> obj1['x'].dtype == 'O'
        True
        """
        copy = self.copy()
        d1 = self.to_dict()
        d2 = dict()
        for k, v in d1.items():
            if type(v) == np.ndarray:
                if v.dtype == 'O':
                    d2[k] = v.astype('str')
                else:
                    d2[k] = v
            else:
                d2[k] = v
        copy.__dict__.update(d2)
        return copy

    def pprint(self, indent=4):
        """
        Tentative at a pretty print

        Parameters
        ----------
        indent : int, optional, default=4
            the number of characters for indent

        Returns
        -------
            screen output of the content of the object
            in a 'pretty' format
        """
        d = self.to_dict()
        pp = pprint.PrettyPrinter(indent=indent)
        pp.pprint(d)

    def to_pandas(self):
        """
        Convert a class object into a pandas dataframe (object)

        Notes
        -----
        Pandas Dataframe does not support units. Use an astropy
        QTable instead

        Examples
        --------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0 ,4])
        >>> df = obj1.to_pandas()
        """
        return pd.DataFrame(self.to_dict())

    def drop(self, iarray, inplace=True):
        """
        Drop an array of rows

        Parameters
        ----------
        iarray : numpy array or list of int
            the list/array of rows to be removed

        inplace : bool, optional, default = True
            True: perform tha task on self
            False: the result is return as a new object

        Returns
        -------
        obj: `class`
            new object of the same class of the input object with the rows
            iarray removed

        >>> import astropy.units as u
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, x, y, z):
        ...         self.x = x
        ...         self.y = y
        ...         self.z = z
        >>> obj1 = Test([2, 1, 5, 6, 5, 3], [2, 5, 6, 8, 0, 9],
        ...             [0 ,4, 5, 9, 1, 2])
        >>> obj2 = obj1.drop([1, 4], inplace=False)
        >>> obj3 = obj1.drop([4], inplace=False)
        >>> obj1 = Test([1, 2, 3, 4, 5], ['a', 'b', 'c', 'd', 'e'],
        ...             [11., 12., 13., 14., 15.] * u.mm)
        >>> obj4 = obj1.drop([4], inplace=False)
        >>> obj4['x']
        array([1., 2., 3., 4.])
        """
        mssg = "Operation cancelled"

        if isinstance(iarray, int):
            ind = np.array([iarray])
        else:
            ind = np.array(iarray)

        cancelled = False
        if self.check_consistency() is False:
            print("Inconsistent object")
            print(mssg)
            return self
        if ind.max() > len(self) - 1:
            print("Index exceeds array max index")
            cancelled = True
        if ind.min() < 0:
            print("Negative index")
            cancelled = True
        if cancelled:
            print(mssg)
            return self

        obj = self[:ind[0]]
        for i in range(1, len(ind)):
            obj += self[ind[i - 1] + 1:ind[i]]
        obj += self[ind[-1] + 1:]
        if inplace:
            self.__dict__.update(obj.__dict__)
        else:
            return obj

    def drop_one(self, i, inplace=True):
        """
        Drop an entry (row index) either with a new object as output or inplace

        Parameters
        ----------
        i : int
            the index of the row to be removed

        inplace : bool, optional, default = True
            True: perform tha task on self
            False: the result is return as a new object

        Returns
        -------
        obj: `class`
            new object of the same class of the input object with the row i
            removed

        Example
        -------
        >>> import numpy as np
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, x, y, z):
        ...         self.x = x
        ...         self.y = y
        ...         self.z = z
        >>> obj1 = Test(['a', 1], 2, [0 ,4, 5])
        >>> obj2 = Test(['a', 'b', 'c'], [0 ,4, 5] , [[1, 2], [3, 4], [5, 6]])
        >>> # obj1.drop(0)  # to drop the first row
        # SizeError: The object has multiple value lengths
        >>> obj2.drop(1)
        >>> all(obj2['x'] == np.array(['a', 'c']))
        True
        >>> all(obj2['y'] == np.array([0, 5]))
        True
        >>> all(obj2['z'][0] == np.array([1, 2]))
        True
        >>> all(obj2['z'][1] == np.array([5, 6]))
        True
        """
        n = self._nelements()
        if isinstance(n, list):
            raise SizeError("The object has multiple value lengths")
        if i > n - 1:
            raise IndexError("Index is greater than the number of rows")
        if i == self._nelements() - 1:
            obj = self[:i]  # a sub part of the sel obj, hence a new on k=j
        elif i == 0:
            obj = self[1:]
        else:
            obj = self[:i] + self[i + 1:]
        if inplace:
            self.__dict__.update(obj.__dict__)
        else:
            return obj

    def methods(self, exclude_dunder=False):
        """
        List the methods of the class for the object

        Parameters
        ----------
        self : the entire object

        Returns
        -------
        A screen display of all the methods associated with the object

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0 ,4])
        >>> 'methods' in obj1.methods()
        True
        """
        # https://stackoverflow.com/
        # questions/1911281/how-do-i-get-list-of-methods-in-a-python-class
        if exclude_dunder:
            return [func for func in dir(self)
                    if callable(getattr(self, func)) and
                    not func.startswith("__")]
        else:
            return [func for func in dir(self)
                    if callable(getattr(self, func))]

    def compare(self, other, order=True):
        """
        Compare two objects

        Parameters
        ----------
        other : an object of a class
            The class should be the same than self but it is not compulsory.

        order : bool, optional, default=True
            if True, the order of the entries matters in the comparison
            if False, any order is valid for a match

        Return
        ------
        added : set
            keys added in addiction to self

        removed : set
            keys removed in the comparison object w.r.t self

        modified: dict
            the modified values

        same: dict
            the same key, value pairs

        Notes
        -----
        The comparison is made at dicitonary level. The keys can be lists
        of lists.
        More complicated constructs are not supported.

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, x, y, z):
        ...         self.x = x
        ...         self.y = y
        ...         self.z = z
        >>> obj1 = Test(['a', 1], 2, [0 ,4, 5])
        >>> obj2 = Test(['a', 0], 2, [0 ,4, 5])
        >>> a, r, m, s, w = obj1.compare(obj1)
        >>> m == {}
        True
        >>> a, r, m, s, w = obj1.compare(obj2)
        >>> m == {'x': (['a', 1], ['a', 0])}
        True
        """
        if not isinstance(other, type(self)):
            warning = "Comparing objects of two different classes"
        else:
            warning = "None"
        added, removed, modified, same = compare_dict(self.to_dict(),
                                                      other.to_dict(),
                                                      order=order)
        return added, removed, modified, same, warning

    def __neq__(self, other):
        """
        Overload the  != operator

        Parameters
        ----------
        other : an object of a class
            The class should be the same than self but it is not compulsory.

        Returns
        -------
        boolean

        Notes
        -----
        It calls the compare method

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0 ,4])
        >>> obj2 = Test(x=['a', 'c'], z=[0 ,3])
        >>> obj1 != obj1
        False
        >>> obj1 != obj2
        True
        """
        added, removed, modified, same, warning = self.compare(other)
        return bool(modified) or bool(added) or bool(removed)

    def __eq__(self, other):
        """
        Overload the  == operator

        Parameters
        ----------
        other : an object of a class
            The class should be the same than self but it is not compulsory.

        Returns
        -------
        boolean

        Notes
        -----
        It calls the compare method

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0 ,4])
        >>> obj2 = Test(x=['a', 'c'], z=[0 ,3])
        >>> obj1 == obj1
        True
        >>> obj1 == obj2
        False
        """
        added, removed, modified, same, warning = self.compare(other)
        return not (bool(added) or bool(removed) or bool(modified))

    def type(self):
        """
        Print the type(values)

        Returns
        -------
            the type of the values in the object on the screen

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0 ,4])
        >>> obj1.type()
        x:  <class 'list'>
        z:  <class 'list'>
        """
        d = self.attribute_type()
        for key, valtype in d.items():
            print(key + ': ', valtype)

    def select_columns(self, keys):
        """
        Select a few columns from an object

        Paraameters
        -----------
        keys : list or str
            a list of keys to be selected

        Returns
        -------
        an object of the same type as self with the selected attributes (keys)

        Examples
        --------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], y = ['dd', 'ee'], z=[0 ,4])
        >>> obj2 = obj1.select_columns(['x', 'z'])
        >>> obj2
        {'x': ['a', 'c'], 'z': [0, 4]}
        >>> obj2 = obj1.select_columns('x')
        >>> obj2
        {'x': ['a', 'c']}
        """
        d = self.to_dict()
        if isinstance(keys, str):
            keys = [keys]
        d2 = {key: value for key, value in d.items() if key in keys}
        v2 = self.__class__()
        v2.__dict__.update(d2)
        return v2

    def pop(self, key):
        """
        Synonym for remove_column (similar syntax for dict)

        Examples
        --------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'x': [['a', 1], ['b', 2]], 'y': [0, 4], 'z': ['r' ,'x' ,'qb']}
        >>> obj1 = Test(**d)
        >>> obj1.pop('z')  # to remove the column z
        >>> obj1
        {'x': [['a', 1], ['b', 2]], 'y': [0, 4]}
        >>> obj1 = Test(**d)
        >>> obj1.pop(['x', 'z'])
        >>> obj1
        {'y': [0, 4]}
 
        """
        self.remove_column(key)

    def remove_column(self, key):
        """
        Remove a column by key

        First the object is transformed into a dictionary,
        then the pop method is used.

        Parameters
        ----------
        key : str or list/np.array of strings
            the key/list of keys to be removed

        Examples
        --------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'x': [['a', 1], ['b', 2]], 'y': [0, 4], 'z': ['r' ,'x' ,'qb']}
        >>> obj1 = Test(**d)
        >>> obj1.remove_column('z')  # to remove the column z
        >>> obj1
        {'x': [['a', 1], ['b', 2]], 'y': [0, 4]}
        >>> obj1 = Test(**d)
        >>> obj1.remove_column(['x', 'z'])
        >>> obj1
        {'y': [0, 4]}
        """
        d = self.to_dict()
        if isinstance(key, (list, np.ndarray, tuple)):
            for k in key:
                d.pop(k, None)
        else:
            d.pop(key, None)
        self.__dict__.update(d)

    def __delitem__(self, key):
        """
        Defines behavior for when an item is deleted (e.g. del self[key]).
        It is a synonym of the remove_column method

        Parameters
        ----------
        key : str
            the key to be removed

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0 ,4])
        >>> del obj1['z']
        >>> obj1
        {'x': ['a', 'c']}
        """
        self.remove_column(key)

    @classmethod
    def from_dict(cls, in_dict, convert=None):
        """
        Create a classUtils object from a dictionary

        in_dict : dict
            the input dictionary

        convert : str, optional, default=None
            if present, convert the Table column objects into
                str='numpy' arrays or str='list'

        Example
        -------
        >>> from classUtils import classUtils
        >>> d = {'a': [1, 2, 3], 'b': ['a', 'b', 'c']}
        >>> obj = classUtils.from_dict(d)
        """
        if not isinstance(in_dict, dict):
            logging.error('The input is not a dictionary')
            return cls()

        if convert is not None:
            for k, v in in_dict.items():
                if isinstance(v, astropy.table.column.Column):
                    if convert == 'numpy':
                        in_dict[k] = np.array(v)
                    elif convert == 'list':
                        in_dict[k] = list(v)
                    # if it is an unknown type, do nothing
        return cls(**in_dict)

    @classmethod
    def from_table(cls, tab, convert=None):
        """
        Instantiate a class from an astropy Table

        Parameters
        ----------
        tab : `astropy.table.table.Table`

        convert : str, optional, default=None
            if present, convert the Table column objects into
                str='numpy' arrays or str='list'

        Notes
        -----
        Have the same restiction as for from_dict

        Example
        -------
        >>> import astropy.units as u
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'x': [['a', 1], ['b', 2]], 'y': [0, 4]}
        >>> obj1 = Test(**d)
        >>> tab1 = obj1.to_table()
        >>> obj2 = Test.from_table(tab1, convert='list')
        >>> d = {'x': [['a', 1], ['b', 2]], 'y': [0, 4] * u.deg}
        >>> obj1 = Test(**d)
        >>> tab1 = obj1.to_table()
        >>> obj2 = Test.from_table(tab1)
        >>> obj1 == obj2
        True
        """
        # Convert first to a dictioanry then call from_dict
        return cls.from_dict(dict(tab), convert=convert)

    def save(self, filename, table=False, format=None,
             overwrite=False, **kwargs):
        """
        Save the object in a file.
        The fits table format is preferred.

        It does not work if there is only one row

        Parameters
        ----------
        filename : str
            the path + filename of the astropy table
            the extension can be ftis, ecvs, ascii or any format accepted
            by astropy.

        table : bool
            if true the astropy table is also provided

        format : str
            the same meaning than for astropy table.write

        overwrite : bool
            True if one wants to force the overwrite exiting file
            with the same name

        **kwargs : keyword argument(s)
            arguments accepted by the astropy table write method

        Examples
        --------
        >>> import numpy as np
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0, 4])
        >>> obj1.save('Test.fits', overwrite='True')
        >>> obj2 = Test(x=np.array(['a', 'c ']).astype('object'), z=[0, 4])
        >>> obj2.save('Test.fits', overwrite='True')
        """
        for k in self.__dict__.keys():
            if k == 'hourAngle':
                self.hourAngle = self.hourAngle.to(u.deg)
        tab = self.object_to_str().to_table()
        tab.write(filename, format=format,
                  overwrite=overwrite, **kwargs)
        if table:
            return tab

    @classmethod
    def from_fits(cls, filename, table=False):
        """
        Wrapper call to from file
        """
        obj = cls.from_file(filename, table=table)
        for k in obj.__dict__.keys():
            if k == 'hourAngle':
                obj.hourAngle = obj.hourAngle.to(u.hourangle)
        return obj

    @classmethod
    def from_file(cls, filename, table=False):
        """
        Read an object from a fits Table created by save

        Usage:
        1) first the fits file should contain one single fits table
        2) the class of the object where we want to read the table into
           ex: from classUtils import classUtils
           All classes inheritated from classUtils can be used
        3) object = name_of_the_class.from_file('filename.fits')
           object = classUtils('filename.fits')

        Parameter
        ---------
        filename : str
            the path + filename of the astropy table fits file

        table : bool, optional, default=False
            return the QTable and the object.
            The call is: tab, obj = cls.from_file(filename)

        Example
        -------
        >>> import astropy.units as u
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'x': [['a', 1], ['b', 2]], 'y': [0, 4]}
        >>> obj1 = Test(**d)
        >>> obj1.save('test.fits', overwrite='True')
        >>> obj2 = Test.from_file('test.fits')
        >>> obj1 == obj2
        True
        """
        tab = QTable.read(filename)
        if table:
            return tab, cls.from_table(tab)
        else:
            return cls.from_table(tab)

    @classmethod
    def from_pandas(cls, df):
        """
        Instantiate a class from a pandas dataframe

        Parameters
        ----------
        df : `pandas.DataFrame`

        Notes
        -----
        No units support from Pandas. Use the Pandas
        function to_dict from a DataFrame

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0 ,4])
        >>> df1 = obj1.to_pandas()
        >>> obj2 = Test.from_pandas(df1)
        >>> obj1 == obj2
        True

        Notes
        -----
        Pandas does not support astropy Quantities
        """
        d = df.to_dict('list')
        for keys, value in d.items():
            d[keys] = np.array(value)
        return cls(**d)

    def rename_column(self, col_dict):
        """
        Rename a key/column name

        Parameter
        ---------
        col_dict : dict
            a dictionary with entries {'old name' : 'new name'}

        Returns
        -------
        Updated object with the requested column renamed. If the
        requested column name does not exist in the objecy, nothing
        is done

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'x': [['a', 1], ['b', 2]], 'y': [0, 4]}
        >>> obj1 = Test(**d)
        >>> obj1.rename_column({'y' : 'z'})
        >>> obj1
        {'x': [['a', 1], ['b', 2]], 'z': [0, 4]}
        >>> obj1 = Test(**d)
        >>> obj1.rename_column({'x' : 'a', 'y' : 'b'})
        >>> obj1
        {'a': [['a', 1], ['b', 2]], 'b': [0, 4]}
        >>> obj1 = Test(**d)
        >>> obj1.rename_column({'x' : 'a', 'z' : 'b'})
        >>> obj1
        {'y': [0, 4], 'a': [['a', 1], ['b', 2]]}
        """
        d = self.to_dict()
        for kold, knew in col_dict.items():
            if kold in d.keys() and knew not in d.keys():
                d[knew] = d.pop(kold, None)
        self.__dict__.update(d)

    def to_hdu(self, remove_inconsistency=False,
               meta=None, filename=None, **kwargs):
        """
        Transform a class object into a FITS BinTableHDU

        Parameters
        ----------
        remove_inconsistency : bool, optional, default=False
            Remove columns with length different than the
            majority.
            Only keep the columns with the full length
            Without this the code will fail to make a table

        meta : dictionary, optional, default = None
            the extra meta data to be added to the HDU header
            with the dictionary keys and values

        filename : str, optional, default = None
            the filename if an output is requested. The extension
            has to be one of the official fits extensions
            (e.g., fit, fits). The filename is the full path.

        **kwargs : any keyword arguments compatible with
            astropy writeto(fileobj, output_verify='exception',
            overwrite=False, checksum=False)

        Returns
        -------
        hdu : fits HDU object
            the fits BinTableHDU

        file on the path : optional if filename is provided

        Example
        -------
        >>> import astropy.units as u
        >>> from astropy.io import fits
        >>> from astropy.table import QTable
        >>> from classUtils import classUtils
        >>> from classUtils import compare_dict
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'x': [1, 2] * u.deg, 'y': [0, 4] * u.m}
        >>> meta = {'ID': 123, 'EXTRA': 'NO'}
        >>> obj = Test(**d)
        >>> filename = 'test_hdu.fits'
        >>> hdu = obj.to_hdu(meta=meta, filename=filename,
        ...                  checksum=True, overwrite=True)
        >>> # Test the output header metadata
        >>> hdul = fits.open(filename)
        >>> hdul[1].header == hdu.header
        True
        >>> # looking at the data
        >>> tab = QTable.read(filename)
        >>> d2 = dict(tab)
        >>> added, removed, modified, same = compare_dict(d, d2)
        >>> modified == {}
        True
        """
        tab = self.to_table(remove_inconsistency=remove_inconsistency)
        hdu = fits.table_to_hdu(tab)
        if meta:
            for key, value in meta.items():
                hdu.header[key] = value
        if filename is not None:
            hdu.writeto(filename, **kwargs)
        return hdu

    def doc(self):
        """
        Print the docstring of the object

        Parameters
        ----------
        self : the object

        Returns
        -------
        Screen display of the docstring

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0 ,4])
        >>> obj1.doc()
        None
        """
        print(inspect.getdoc(self))

    def stack(self, other):
        """
        Stack two objects instead of purely adding the
        attributes

        A wrapper to __add__

        Parameters
        ----------
        self : the object

        other : an object of a class
            The class should be the same than self but it is not compulsory.

        Returns
        -------
        self:
            stacked object between self and other

        Example
        -------
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> obj1 = Test(x=['a', 'c'], z=[0, 4])
        >>> obj2 = Test(x=['b', 'd'], z=[1, 2])
        >>> obj3 = obj1 + obj2
        >>> obj5 = obj1.stack(obj2)
        >>> obj5['x'][0]
        array(['a', 'c'], dtype=object)
        >>> obj5['x'][1]
        array(['b', 'd'], dtype=object)
        """
        return self.__add__(other, stack=True)

    def to_numpy(self):
        """
        Transform all the quantities into pure values/numpy arrays

        It uses the routine for dictionary to_numpy

        The keys and units are provided as separate dictonnaries

        Parameter
        ---------
        self : an object of the class

        Returns
        -------
        dtype : dict
            the keys are the keys of the input dict
            the values are the type of each key

        dunits : dict
            the keys are the keys of the input dict
            the values are the units of each key

        d2 : an object of the same class as self
            the same values as the self but without the units

        Example
        -------
        >>> import numpy as np
        >>> import astropy.units as u
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'a': [1, 2] * u.mm, 'b': 3 * u.deg, 'c': np.array([4, 5])}
        >>> obj1 = Test(**d)
        >>> dtype, dunits, obj2 = obj1.to_numpy()
        >>> print(all(obj2.a == np.array([1, 2])))
        True
        """
        copy = self.copy()
        d = copy.to_dict()
        dtype, dunits, d2 = to_numpy(d)
        copy.__dict__.update(d2)
        return dtype, dunits, copy

    def to_csv(self, filename, append=False):
        """
        Write to a csv file.

        https://www.tutorialspoint.com/
        How-to-save-a-Python-Dictionary-to-CSV-file

        It is better to use directly the write to ecsv if the
        table is conformal

        Parameter
        --------
        filename : str
            the full path the csv file

        append : bool, optional, default = False
            append to the file filename instead of creating a new one

        Example
        -------
        >>> import numpy as np
        >>> import astropy.units as u
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'x': [1, 2] * u.deg, 'y': [0, 4] * u.m}
        >>> obj = Test(**d)
        >>> obj2 = Test()
        >>> filename = "text_csv.txt"
        >>> obj.to_csv(filename)
        >>> obj2 = Test.from_csv(filename)
        >>> obj == obj2
        True
        """
        # Remove the units and save the information
        dtype, dunits, obj = self.to_numpy()

        nsize = obj._nelements()
        if isinstance(nsize, list):
            lobj = nsize[0]
        else:
            lobj = nsize

        # append mode = a, write mode = w
        if append:
            mode = 'a'
            dict_data = []
        else:
            mode = 'w'
            # a new file will have 2 extra lines
            # for the header
            dict_data = [dtype]  # type
            dict_data.append(dunits)  # units

        csv_columns = list(dunits.keys())

        # create the rows
        if lobj > 1:
            for row in obj:
                dict_data.append(row.__dict__)
        else:
            dict_data.append(obj.__dict__)

        try:
            with open(filename, mode) as csvfile:
                writer = csv.DictWriter(csvfile,
                                        delimiter=',',
                                        fieldnames=csv_columns,
                                        quoting=csv.QUOTE_NONNUMERIC)
                if not append:
                    writer.writeheader()
                for data in dict_data:
                    writer.writerow(data)
        except IOError:
            print("I/O error")

    @classmethod
    def from_csv(cls, filename, variable_type=True, variable_units=True,
                 delimiter=','):
        """
        Read an object of class classUtils or children class of it
        from a csv file

        csv format file
        Row 1: header = keys
        Row 2: type of the variables
        Row 2: units, to be able to reconstruct Quantities
        Row 3 ...: the actual data

        Notes
        -----
        CSV data are read as string. One needs to combine with
        the separate type and units information to reconstruct
        the initial data.

        The reader requires at least that a header with the name of the
        columns is provided.

        Parameter
        ---------
        filename : str
            the full path the csv file

        variable_type : boolean, optional, default=True
            Do the data include information about the type (float, int, ...)
            for the data? The user needs to look at the format of the data
            first. 

        variable_units : boolean, optional, default=True
            Do the data include include information about the units?
            The user needs to look at the format of the data first. 

        delimiter : str, optional, default=','
            The delimiter used to separate the data
        
        Example
        -------
        >>> import numpy as np
        >>> import astropy.units as u
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d = {'a': [1, 2] * u.mm, 'b': [3., 45.] * u.deg,
        ...      'c': np.array([4, 5])}
        >>> v1 = Test.from_dict(d)
        >>> v1.to_csv('test.csv')
        >>> v2 = Test.from_csv('test.csv')
        >>> v1 == v2
        True
        >>> d = {'a': [1, 2] * u.mm, 'b': [3., 45.] * u.deg,
        ...      'c': np.array(['b', 'a a '])}
        >>> v1 = Test.from_dict(d)
        >>> v1.to_csv('test.csv')
        >>> v2 = Test.from_csv('test.csv')
        """
        file = open(filename)
        try:
            csvreader = csv.reader(file, delimiter=delimiter,
                               quoting=csv.QUOTE_NONNUMERIC)
            keys = next(csvreader)
        except ValueError:  # Header entries have no quote
            file.close()
            file = open(filename)
            csvreader = csv.reader(file, delimiter=delimiter)
            keys = next(csvreader)
        nb_columns = len(keys)
        if variable_type:
            types = next(csvreader)
        else:
            types = [''] * nb_columns
        if variable_units:   
            units = next(csvreader)
        else:
            units = [''] * nb_columns

        # Read the csv file row by row
        rows = []
        for ir, row in enumerate(csvreader):
            rows.append(row)

        # convert the row by row data into columns
        if ir > 0:  # transpose of a list
            col = list(map(list, zip(*rows)))
        else:
            col = row
        d = dict()

        # interprete the data with their type and units
        for k1, t1, u1, c1 in zip(keys, types, units, col):
            d[k1] = np.array(c1)
            llist = False
            if ir == 0:  # one row
                if ('[' not in c1) and (']' not in c1):
                    if 'float' in t1:  # assuming simple types
                        d[k1] = float(c1)
                    elif 'int' in t1:
                        d[k1] = int(c1)
                    elif 'bool' in t1:
                        d[k1] = bool(c1)
                    else:  # string
                        d[k1] = c1
                else:  # remove extra
                    c2 = c1.replace(']', '').replace('[', '').replace("'", '')
                    c2 = c2.replace('"', '').split(' ')
                    d[k1] = np.array(c2).astype(t1)
            else:  # multiple rows
                try:
                    # convert the list to a numpy array with the right type
                    NoneExist = False
                    if 'bool' not in t1:  # t1 is the type
                        if '' in c1:  # There are at least one None value
                            type_str = False
                            for x in c1:
                                if isinstance(x, str):
                                    type_str = True
                            if not type_str:
                                d[k1] = [None if x == '' else x for x in c1]
                            NoneExist = True
                        else:
                            if 'int' in t1:  # pb with numpy.int64, ...
                                d[k1] = np.array(c1).astype('int')
                            elif 'float' in t1:
                                d[k1] = np.array(c1).astype('float')
                            elif 'str' in t1:
                                d[k1] = np.array(c1).astype('str')
                            elif 'bool' in t1:
                                d[k1] = np.array(c1).astype('bool')
                            else:
                                d[k1] = np.array(c1)
                    else:  # there is no specific type for bool
                        d[k1] = np.array(c1)
                except ValueError:  # inhomogeneous data
                    c4 = []
                    for c2 in c1:
                        c2 = c2.replace(']', '').replace('[', '')
                        c2 = c2.replace("'", '')
                        c2 = c2.replace('"', '').split(' ')
                        if 'bool' not in t1:  # again the same issue for bool
                            c3 = np.array(c2).astype(t1)
                        else:
                            c3 = np.array(c2)
                        if u1 != '':  # put back the units
                            c3 *= u.Unit(u1)
                        c4.append(c3)
                        llist = True
                    d[k1] = c4  #
            if u1 != '' and not llist:  # assign a units
                if NoneExist:
                    v2 = []
                    for v in d[k1]:
                        if v != '' and v is not None:
                            v2.append(v * u.Unit(u1))
                        else:
                            v2.append(v)
                    d[k1] = v2
                else:
                    d[k1] *= u.Unit(u1)
        #  return the class from the dictionary
        return cls(**d)

    def pivot(self, new_key, pivot_name=None):
        """
        Pivot the object by adding a pivot 'column', which have the
        input keys as values

        Parameter
        ---------
        self : classUtils object
            an instance of the class

        new_key : str
            the name of one of the keys whose unique values will become the
            new header/column names

        pivot_name : str, optional, default=None
            the name of the new pivot

        Examples
        --------
        >>> from classUtils import classUtils
        >>> class Car(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> dealer1_dict = {'Mercedes': 2, 'Toyota': 4, 'BMW': 0}
        >>> dealer2_dict = {'Mercedes': 1, 'Toyota': 2, 'BMW': 1}
        >>> dealer1 = Car(**dealer1_dict)
        >>> dealer1.name = 'Jane'
        >>> dealer2 = Car(**dealer2_dict)
        >>> dealer2.name = 'John'
        >>> dealers = dealer1 + dealer2
        >>> new_obj = dealers.pivot('name', pivot_name='Brand')
        >>> dealers2 = new_obj.pivot('Brand', pivot_name='name')
        >>> dealers == dealers2
        True
        """
        if new_key not in self.keys():
            logging.error(f"New key is not one of the keys: {self.keys()}")
            return self
        ll = [v for k, v in self.items() if k != new_key]
        # tranpose of a list of lists
        trll = list(map(list, zip(*ll)))
        new_dict = {}
        for k, v in zip(self[new_key], trll):
            new_dict[k] = v
        if pivot_name is None:
            pivot_name = 'pivot'
        pivot = self.keys()
        pivot.remove(new_key)
        new_dict[pivot_name] = pivot
        return classUtils(**new_dict)

    def merge(self, other, inplace=False, **kwargs):
        """
        Parameter
        ---------
        self : classUtils class
            The first object

        self : classUtils class
            The second object

        merge_as_list : boolean, optional, default=False
            ignore quantities and numpy array and merge the values
            as lists

        no_units : boolean, optional, default=False
            return only numpy arrays

        same_value_type : boolean, optional, default=False
            can only merge lists with values of the same type

        verbose : boolean, optional, default=False
            Output error messages if set to True

        Returns
        -------
        self : classUtils class
            if inplace=True, self mergerd with other

        : classUtils class
            merged objects as a new object

        Examples
        --------
        >>> import astropy.units as u
        >>> from classUtils import classUtils
        >>> class Test(classUtils):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        >>> d1 = {'a': [1, 2, 3], 'b': ['g', 'f', 'e']}
        >>> d2 = {'a': [132, 21, 31], 'b': ['h', 'i', 'j'],
        ...       'c': [3, 4, 0]}
        >>> test1 = Test.from_dict(d1)
        >>> test2 = Test.from_dict(d2)
        >>> md = test1.merge(test2)
        >>> md.to_csv('test_merge.csv')
        >>> md2 = Test.from_csv('test_merge.csv')
        >>> md == md2
        True
        >>> d1 = {'a': [1, 2, 3] * u.cm, 'b': ['g', 'f', 'e']}
        >>> d2 = {'a': [132, 21, 31] * u.cm, 'b': ['h', 'i', 'j'],
        ...       'c': [3, 4, 0]}
        >>> test1 = Test.from_dict(d1)
        >>> test2 = Test.from_dict(d2)
        >>> md = test1.merge(test2)
        >>> md.to_csv('test_merge.csv')
        >>> md2 = Test.from_csv('test_merge.csv')
        >>> md == md2
        True
        >>> d1 = {'a': [1, 2, 3] * u.cm, 'b': ['g', 'f', 'e']}
        >>> d2 = {'a': [132, 21, 31] * u.cm, 'b': ['h', 'i', 'j'],
        ...       'c': [3, 4, 0] * u.K}
        >>> test1 = Test.from_dict(d1)
        >>> test2 = Test.from_dict(d2)
        >>> md = test1.merge(test2)
        >>> md.to_csv('test_merge.csv')
        >>> md2 = Test.from_csv('test_merge.csv')
        >>> md == md2
        True
        """
        d1 = self.to_dict()
        d2 = other.to_dict()
        md = merge_dict(d1, d2, **kwargs)
        if inplace:
            self.__dict__.update(md)
            return self
        else:
            out = self.copy()
            out.__dict__.update(md)
            return out


class SizeError(Exception):
    pass


if __name__ == "__main__":
    import doctest
    DOCTEST = True
    doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)

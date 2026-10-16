# classUtils – Scientific Container Class

> A dictionary-like Python object purpose-built for **astronomy / physics
> data workflows**, with first-class support for `numpy` arrays,
> `astropy` Quantities (values + units), `pandas` DataFrames, and
> FITS / ECSV files.

**Author:** Wing-Fai Thi
**License:** [GPL-3.0](https://www.gnu.org/licenses/)
**Python:** ≥ 3.9

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation & Dependencies](#installation--dependencies)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Objects as enhanced dictionaries](#objects-as-enhanced-dictionaries)
  - [Row operations](#row-operations)
  - [Column operations](#column-operations)
  - [Unit-aware data](#unit-aware-data)
  - [Consistency checks & decorators](#consistency-checks--decorators)
- [API Reference – Module-Level Functions](#api-reference--module-level-functions)
- [API Reference – `classUtils` Class](#api-reference--classutils-class)
- [Data Conversion Matrix](#data-conversion-matrix)
- [File I/O](#file-io)
- [Plotting](#plotting)
- [Design Decisions & Patterns](#design-decisions--patterns)
- [Known Limitations](#known-limitations)
- [Bug Fixes Applied (v1.4)](#bug-fixes-applied-v14)
- [Testing](#testing)
- [Changelog](#changelog)
- [Contributing / Extending](#contributing--extending)
- [License](#license)

---

## Overview

`classUtils` is a **base class** you inherit from to turn any collection
of named arrays into a rich, dictionary-like object with scientific-data
superpowers:

| Capability | Example |
|---|---|
| Dict-style access | `obj['x']`, `obj.x`, `obj.keys()` |
| Row indexing / slicing | `obj[0]`, `obj[1:4]`, `obj[[0,2,4]]` |
| Boolean-mask selection | `obj[mask]` |
| Row assignment | `obj[0:3] = other[1:4]` |
| Add / Stack / Merge | `obj1 + obj2`, `obj1.stack(obj2)`, `obj1.merge(obj2)` |
| Sort / Lexsort | `obj.sort('mag')`, `obj.lexsort(['z','x'])` |
| Drop rows / columns | `obj.drop([1,4])`, `del obj['col']` |
| Pivot / Rename | `obj.pivot('brand')`, `obj.rename_column({'a':'b'})` |
| Convert to/from | `to_table()`, `to_pandas()`, `to_numpy()`, `to_hdu()` |
| File I/O | `save()`, `from_fits()`, `from_file()`, `from_csv()`, `to_csv()`, `from_pandas()` |
| Unit-aware (astropy) | Quantities survive round-trips through FITS / ECSV |
| Consistency guards | `@ensure_consistency`, `@auto_pad_consistency`, `check_consistency()` |
| Quick plotting | `obj.plot('ra', 'dec')` |

The class is designed as a **parent class**. A minimal concrete subclass
requires only:

```python
from classUtils import classUtils

class MyClass(classUtils):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
```

---

## Features

- **Zero boilerplate** – inherit and pass `**kwargs`; every kwarg becomes a column.
- **Operator overloading** – `+`, `+=`, `==`, `!=`, `len()`, `del obj['col']`.
- **Unit preservation** – `astropy.units.Quantity` arrays are kept through
  most operations; `to_numpy()` / `from_csv()` can strip and restore them.
- **FITS / ECSV native** – `save()` writes a FITS BinTable; `from_fits()`
  reads it back. `to_hdu()` gives a raw `BinTableHDU` with custom headers.
- **CSV with metadata** – `to_csv()` writes type and unit header rows so
  `from_csv()` can reconstruct Quantities.
- **pandas interop** – `to_pandas()` / `from_pandas()` (no units in Pandas).
- **astropy QTable interop** – `to_table()` / `from_table()` (units preserved).
- **Sorting** – single-key (`sort` / `argsort`) and multi-key
  (`lexsort` / `arglexsort`).
- **Pivot** – reshape a "long" table to "wide" (or vice-versa).
- **Filter / Mask** – `obj.filter(obj.mag < 5)`.
- **Pretty print** – `obj.pprint()`.
- **Deep copy** – `obj.copy()` uses `copy.deepcopy`.
- **Docstring introspection** – `obj.doc()`, `obj.methods()`.

---

## Project Structure

```
ScientificContainerClass/
├── classUtils.py          # ← the module documented here
├── index_all.py           # helper used by to_numpy() to find embedded Quantities
├── README.md              # this file
└── (your concrete classes, tests, data, …)
```

> `classUtils` is designed to be **imported and subclassed**. Concrete
> scientific classes (e.g. `SpectralLine`, `DetectorFrame`) inherit from it.

---

## Installation & Dependencies

### Required Python packages

| Package      | Minimum version | Role                              |
|--------------|-----------------|-----------------------------------|
| `numpy`      | ≥ 1.21          | array engine                      |
| `astropy`    | ≥ 5.0           | `Quantity`, `QTable`, `fits`, `Time` |
| `pandas`     | ≥ 1.3           | DataFrame interop                 |
| `matplotlib` | ≥ 3.5           | quick plotting                    |

### Local / optional

| Module        | Note                                                       |
|---------------|------------------------------------------------------------|
| `index_all`   | local helper (`index_all.py` in the same directory)       |

### Install

```bash
# from the project root
pip install numpy astropy pandas matplotlib

# no pip-installable package; add the directory to your PYTHONPATH
# or use the module in-place:
cd ScientificContainerClass
python -c "from classUtils import classUtils; print('OK')"
```

---

## Quick Start

```python
from classUtils import classUtils
import numpy as np
import astropy.units as u

# ── 1. Define a concrete class (one-line boilerplate) ──────────────
class SpectralLine(classUtils):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# ── 2. Create objects ──────────────────────────────────────────────
line_a = SpectralLine(
    ra   = np.array([10.5, 11.2]) * u.deg,
    dec  = np.array([-2.1,  3.7]) * u.deg,
    flux = np.array([120.4,  98.2]) * u.Jy,
    name = ['Hα', 'Hβ'],
)

line_b = SpectralLine(
    ra   = np.array([12.0]) * u.deg,
    dec  = np.array([ 0.5]) * u.deg,
    flux = np.array([ 75.0]) * u.Jy,
    name = ['OIII'],
)

# ── 3. Combine (append rows) ───────────────────────────────────────
all_lines = line_a + line_b          # 3 rows  (new object)
all_lines += line_b                  # in-place append

# ── 4. Index / slice / mask ────────────────────────────────────────
print(all_lines[0])                  # single row
print(all_lines[0:2])                # slice
bright = all_lines[all_lines['flux'] > 100 * u.Jy]   # boolean mask

# ── 5. Sort ────────────────────────────────────────────────────────
all_lines.sort('flux')               # in-place, ascending

# ── 6. Convert ─────────────────────────────────────────────────────
tab = all_lines.to_table()           # astropy QTable (units preserved)
df  = all_lines.to_pandas()          # pandas DataFrame  (no units)

# ── 7. Save / Load ─────────────────────────────────────────────────
all_lines.save('lines.fits', overwrite=True)
reloaded = SpectralLine.from_fits('lines.fits')
print(all_lines == reloaded)         # True

# ── 8. Quick plot ──────────────────────────────────────────────────
all_lines.plot('ra', 'dec', s=30)
```

---

## Core Concepts

### Objects as enhanced dictionaries

Every keyword argument passed to `__init__` becomes an **attribute**
(column). Access is interchangeable:

```python
obj['ra']       # dict-style
obj.ra          # attribute-style
obj.keys()      # ['ra', 'dec', 'flux', 'name']
obj.values()
obj.items()
```

The object **is** its `__dict__`. `to_dict()` simply returns
`self.__dict__`.

### Row operations

| Syntax               | Meaning                                          |
|----------------------|--------------------------------------------------|
| `obj[0]`            | single row (returns a new object with scalars)   |
| `obj[1:4]`          | slice of rows                                    |
| `obj[[0, 2, 4]]`    | fancy indexing (list of indices)                 |
| `obj[(0, 2)]`       | tuple of indices                                 |
| `obj[mask]`         | boolean-mask selection                           |
| `obj[0:3] = other[1:4]` | **assign** rows from another object          |

> **Note:** `len(obj)` returns the **maximum** column length (always an
> `int`). Use `obj._nelements()` to see per-column lengths when they
> disagree.

### Column operations

| Method                          | Description                                      |
|---------------------------------|--------------------------------------------------|
| `obj1 + obj2`                  | append rows (numpy-style `append`) → new object  |
| `obj1 += obj2`                 | in-place append                                  |
| `obj1.stack(obj2)`             | stack rows (preserves sub-array structure)       |
| `obj1.merge(obj2, **kw)`       | merge with optional unit / type checks           |
| `obj.sort('key')`              | single-key sort (in-place by default)            |
| `obj.lexsort(['a','b'])`       | multi-key lexicographic sort                     |
| `obj.drop([1,4])`              | remove rows by index                             |
| `obj.drop_one(0)`              | remove a single row                              |
| `obj.remove_column('x')`       | remove a column                                  |
| `del obj['x']`                 | same as `remove_column`                          |
| `obj.rename_column({'a':'b'})` | rename column(s)                                 |
| `obj.select_columns(['x','z'])`| keep only listed columns                         |
| `obj.pivot('name')`            | pivot (long ↔ wide)                              |
| `obj.filter(mask)`             | boolean-mask filter (decorated)                  |

### Unit-aware data

`astropy.units.Quantity` arrays are first-class citizens:

```python
obj['ra'] = np.array([1.0, 2.0]) * u.deg
obj['ra'].unit          # <Unit("deg")>
obj['ra'].value         # array([1.0, 2.0])
```

- `to_numpy()` (class method) strips units → returns
  `(dtype_dict, units_dict, plain_obj)`.
- The **module-level** `to_numpy(d)` takes a plain dict and does the same
  conversion without making a copy.
- `add_dict` / `stack_dict` **refuse** to mix incompatible units
  (returns `{}`) unless you use `stack=True`.
- `to_table()` / `to_hdu()` / `save()` preserve units in FITS / ECSV.
- `to_pandas()` **cannot** preserve units (pandas limitation).

### Consistency checks & decorators

| Tool                              | Purpose                                          |
|-----------------------------------|--------------------------------------------------|
| `obj.check_consistency()`         | verify all columns have the same length          |
| `obj.check_column_type()`         | verify columns are `list` / `ndarray` / `Quantity` |
| `@ensure_consistency`            | decorator: raise `ValueError` if column lengths diverge after a method |
| `@auto_pad_consistency`          | decorator: pad shorter columns with `NaN` / `None` |
| `obj._nelements(exclude=[…])`    | per-column length (int if uniform, list otherwise) |

---

## API Reference – Module-Level Functions

### `constant(f)`

Decorator that turns a zero-argument method into a **read-only property**.
Attempting assignment raises `TypeError("Constant cannot be changed")`.

```python
class Foo:
    @constant
    def pi(self):
        return 3.14159

f = Foo()
f.pi          # 3.14159
f.pi = 1.0    # TypeError
```

---

### `to_numpy(d: dict) → (dtype: dict, dunits: dict, d2: dict)`

**Module-level function.** Strips `astropy` units from every value in a
plain dictionary.

| Return | Description |
|--------|-------------|
| `dtype`  | dict of key → element type string (e.g. `'float64'`, `'int64'`) |
| `dunits` | dict of key → `astropy.units.Unit` or `''` |
| `d2`     | dict of key → unit-free value (list / ndarray / scalar) |

```python
>>> d = {'a': [1, 2] * u.mm, 'b': 3 * u.deg, 'c': np.array([4, 5])}
>>> dtype, dunits, obj = to_numpy(d)
>>> dtype
{'a': 'float64', 'b': 'float64', 'c': 'int64'}
>>> dunits
{'a': Unit("mm"), 'b': Unit("deg"), 'c': ''}
>>> obj
{'a': array([1., 2.]), 'b': 3.0, 'c': array([4, 5])}
```

> Handles nested lists containing a mix of Quantities and plain values,
> including `None` / `''` placeholders.

---

### `len_dict(d: dict) → int | list[int]`

Length of each value. Returns a single `int` if all columns have the
same length, otherwise a `list` of the distinct lengths.

```python
>>> len_dict({'a': [2, 3], 'b': ['c', 'd', 'e']})
[2, 3]
```

---

### `add_dict(d1, d2, stack=False) → dict`

Append `d2`'s values to `d1`'s (numpy `append`). With `stack=True`,
arrays of equal shape are **stacked** (`np.vstack`) instead.

- Returns `{}` on column mismatch or incompatible units.
- Logs a warning and suggests `stack=True` when units differ.

```python
>>> d1 = {'a': np.array([1, 2]), 'b': 'b'}
>>> d2 = {'a': 4, 'b': 'v'}
>>> add_dict(d1, d2)
{'a': array([1, 2, 4]), 'b': array(['b', 'v'], dtype='<U1')}
```

---

### `use_list(dd: dict, only_string=True) → dict`

Convert `ndarray` columns to plain `list`. By default only string arrays
are converted; set `only_string=False` to convert all numeric arrays too.

```python
>>> dd = {'a': np.array([1., 2.]), 'c': np.array(['xx', 'yy'])}
>>> use_list(dd)                        # only string arrays
{'a': array([1., 2.]), 'c': ['xx', 'yy']}
>>> use_list(dd, only_string=False)    # all arrays
{'a': [1.0, 2.0], 'c': ['xx', 'yy']}
```

---

### `stack_dict(d1, d2) → dict`

Internal helper: stack two dictionaries row-wise (transpose).
Preserves units when both dicts share the same unit per key.
Called automatically by `add_dict(..., stack=True)`.

---

### `compare_dict(d1, d2, order=True) → (added, removed, modified, same)`

Set-diff of keys + per-key value comparison.

| Return    | Type  | Description                              |
|-----------|-------|------------------------------------------|
| `added`   | `set` | keys in `d1` not in `d2`                |
| `removed` | `set` | keys in `d2` not in `d1`                |
| `modified`| `dict`| key → `(d1_value, d2_value)` pairs      |
| `same`    | `dict`| key → value (identical in both)         |

`order=False` makes the comparison order-insensitive (sorts arrays first).
Handles numpy array-of-boolean comparison (avoids `FutureWarning`).

---

### `merge_dict(d1, d2, merge_as_list=False, no_units=False, same_value_type=False, verbose=False) → dict | None`

Merge two dicts with optional checks. Returns `None` on failure.

| Parameter         | Default | Description |
|-------------------|---------|-------------|
| `merge_as_list`   | `False` | ignore Quantities, return plain lists |
| `no_units`        | `False` | return numpy arrays without Quantities |
| `same_value_type` | `False` | abort if per-key dtypes differ |
| `verbose`         | `False` | log error details |

Requires all values within each dict to have the same length.
Keys present in only one dict are padded with `None`.

---

### `check_unique(mydict, key_check) → bool`

Return `False` if any key in `key_check` contains duplicate values.
Logs info messages for each key checked.

---

### `ensure_consistency(f)`

Decorator: after `f` runs, asserts all column lengths match.
Raises `ValueError` on mismatch.

```python
class MyClass(classUtils):
    @ensure_consistency
    def add_column(self, name, values):
        setattr(self, name, values)
```

---

### `auto_pad_consistency(f)`

Decorator: after `f` runs, pads shorter columns to the max length with
`NaN` (numeric arrays / Quantities) or `None` (object columns).

```python
class MyClass(classUtils):
    @auto_pad_consistency
    def add_short_column(self, name, values):
        setattr(self, name, values)   # auto-padded to match
```

---

### `SizeError`

Custom exception (inherits `Exception`). Raised by `drop_one` when
column lengths are inconsistent.

---

## API Reference – `classUtils` Class

### Constructor & Dunder Methods

| Method | Description |
|---|---|
| `__init__(self, **kwargs)` | Store every kwarg as an attribute via `self.__dict__.update(kwargs)` |
| `__repr__()` | Returns `str(self.__dict__)` |
| `__len__()` | Returns `self._nelements()` → max column length (int) |
| `__getitem__(item)` | `str` → column; `int`/`slice` → single row; `list`/`ndarray`/`tuple`/`array` → row(s) |
| `__setitem__(key, value)` | `str` → set column; `int`/`slice` → assign rows; `list`/`ndarray` → assign rows by index |
| `__delitem__(key)` | Remove a column (synonym of `remove_column`) |
| `__add__(other, stack=False)` | Append / stack rows → **new** object; returns `[self, other]` on key mismatch |
| `__iadd__(other)` | In-place append (`obj1 += obj2`) |
| `__eq__(other)` | `True` if all columns identical (via `compare_dict`) |
| `__ne__(other)` | Negation of `__eq__` |

### Accessor / Introspection

| Method | Returns |
|---|---|
| `to_dict()` | `dict` (== `self.__dict__`) |
| `keys()` | `list[str]` |
| `values()` | `list` |
| `items()` | `list[tuple[str, Any]]` |
| `keys_to_screen()` | `"<ClassName names=('a','b')>"` |
| `number_columns()` | `int` (alias for `number_attributes`) |
| `number_attributes()` | `int` |
| `attribute_type()` / `info()` | `dict` of key → type (Quantities show unit string) |
| `type()` | Prints key → type to screen |
| `methods(exclude_dunder=False)` | `list[str]` of callable names |
| `doc()` | Prints the class docstring via `inspect.getdoc` |
| `pprint(indent=4)` | Pretty-prints the dict via `pprint.PrettyPrinter` |
| `copy()` | Deep copy via `copy.deepcopy` |
| `_nelements(exclude=[])` | `int` or `list[int]` of per-column lengths |
| `clean_quantities()` | Converts list-of-Quantities → single Quantity array (in-place) |

### Row Operations

| Method | Description |
|---|---|
| `drop(iarray, inplace=True)` | Remove rows by index list. Validates bounds. |
| `drop_one(i, inplace=True)` | Remove a single row. Raises `SizeError` if inconsistent. |
| `filter(mask)` | Boolean-mask subset. Decorated with `@ensure_consistency`. Returns a new object. |

### Sorting

| Method | Description |
|---|---|
| `argsort(key)` | `np.ndarray[int]` – indices to sort by `key` |
| `sort(key, inplace=True)` | Sort rows by `key`. In-place or new object. |
| `arglexsort(keys)` | `np.ndarray[int]` – indices for multi-key lexicographic sort |
| `lexsort(keys, inplace=True)` | Multi-key lexicographic sort. Right-most key is primary. |

> All sort methods check key existence and column consistency, logging
> a warning and returning `None` / `self` on failure.

### Column Operations

| Method | Description |
|---|---|
| `select_columns(keys)` | Keep only listed columns → new object |
| `remove_column(key)` | Remove column by `str` or `list[str]` |
| `pop(key)` | Synonym for `remove_column` |
| `rename_column(col_dict)` | Rename via `{'old': 'new'}`. No-op if target exists. |
| `stack(other)` | Wrapper: `self.__add__(other, stack=True)` |
| `merge(other, inplace=False, **kw)` | Merge with unit / type checks (delegates to `merge_dict`) |
| `pivot(new_key, pivot_name=None)` | Long ↔ wide reshape. Returns a new object of the same class. |

### Consistency & Validation

| Method | Description |
|---|---|
| `check_consistency(remove_inconsistency=False, keep_one=False, verbose=False)` | `True` if all columns same length. Optionally removes inconsistent columns in-place. |
| `check_column_type(verbose=False)` | `True` if all multi-value columns are `list` / `ndarray` / `Quantity` |
| `compare(other, order=True)` | → `(added, removed, modified, same, warning)` |

### Conversion (Object → Other)

| Method | Output | Units preserved? |
|---|---|---|
| `to_dict()` | `dict` | ✔ |
| `to_numpy()` | `(dtype: dict, dunits: dict, obj_no_units)` | stripped (returned separately) |
| `to_table(remove_inconsistency=False)` | `astropy.table.QTable` | ✔ |
| `to_pandas()` | `pandas.DataFrame` | ✘ |
| `to_hdu(meta=None, filename=None, **kw)` | `fits.BinTableHDU` (+ optional file write) | ✔ |
| `to_csv(filename, append=False)` | CSV file (type + unit header rows) | ✔ (stored as strings) |

> **`to_numpy()` is a class method** that takes only `self`. It makes a
> deep copy, calls the **module-level** `to_numpy(d)` on the copy's dict,
> and returns `(dtype, dunits, copy)`. Do not pass an extra argument.

### Conversion (Other → Object)

| Method | Source | Units? |
|---|---|---|
| `from_dict(cls, in_dict, convert=None)` | `dict` | ✔ |
| `from_table(cls, tab, convert=None)` | `astropy QTable` | ✔ |
| `from_pandas(cls, df)` | `pandas.DataFrame` | ✘ |
| `from_file(cls, filename, table=False)` | FITS / ECSV (astropy) | ✔ |
| `from_fits(cls, filename, table=False)` | FITS (alias) | ✔ |
| `from_csv(cls, filename, variable_type=True, variable_units=True, delimiter=',')` | CSV | ✔ (reconstructed) |

All `from_*` methods are **`@classmethod`** constructors.

### File I/O

| Method | Direction | Format | Notes |
|---|---|---|---|
| `save(filename, format=None, overwrite=False, **kw)` | write | FITS / ECSV / ASCII (astropy) | Calls `object_to_str()` first |
| `from_file(filename, table=False)` | read | same as `save` | `table=True` → `(QTable, obj)` |
| `from_fits(filename, table=False)` | read | FITS | Wrapper for `from_file` |
| `to_csv(filename, append=False)` | write | CSV | 3 header rows: keys, types, units |
| `from_csv(filename, …)` | read | CSV | Reconstructs types + units |
| `to_pandas()` / `from_pandas(df)` | both | DataFrame | No units |
| `to_table()` / `from_table(tab)` | both | QTable | Units preserved |

### Plotting

| Method | Description |
|---|---|
| `plot(x_key, y_key, **kwargs)` | Scatter plot via matplotlib. Auto-labels axes with units when the column is a `Quantity`. Calls `self.to_numpy()` (class method, no extra args) to extract units. |

---

## Data Conversion Matrix

```
                     ┌───────────────┐
    kwargs / dict ──►│               │──► to_table()     ──►  QTable
                     │   classUtils  │──► to_pandas()    ──►  DataFrame
    from_dict()      │               │──► to_numpy()     ──►  (dtype, dunits, plain)
    from_table() ◄───│   (object)    │──► to_hdu()       ──►  BinTableHDU
    from_pandas()    │               │──► to_csv()       ──►  .csv
    from_file()      │               │──► to_dict()      ──►  dict
    from_fits()      │               │
    from_csv()       └───────────────┘
```

> **Units** survive every path **except** `to_pandas()` / `from_pandas()`.

---

## File I/O

### FITS / ECSV (preferred)

```python
obj.save('data.fits', overwrite=True)
obj.save('data.ecsv', overwrite=True)

loaded = MySubclass.from_fits('data.fits')
tab, loaded = MySubclass.from_file('data.ecsv', table=True)  # also get QTable
```

### FITS HDU with custom metadata

```python
hdu = obj.to_hdu(
    meta={'OBSERVER': 'JDOE', 'FILTER': 'Hα', 'EXPOSURE': 300},
    filename='obs.fits',
    overwrite=True,
)
# hdu.header['OBSERVER']  →  'JDOE'
```

### CSV (with type & unit header)

```python
obj.to_csv('export.csv')
reloaded = MySubclass.from_csv('export.csv')
```

File layout:

```
Row 1:  column names (keys)
Row 2:  numpy dtype strings   (e.g. float64, int64, str)
Row 3:  astropy unit strings  (e.g. deg, mm, '' for dimensionless)
Row 4+: data
```

`from_csv` parameters let you skip the type or units row if your file
was not produced by `to_csv`:

```python
MySubclass.from_csv('external.csv', variable_type=False, variable_units=False)
```

### Pandas

```python
df   = obj.to_pandas()            # units lost
obj2 = MySubclass.from_pandas(df)
```

---

## Plotting

```python
obj.plot('ra', 'dec', s=50, alpha=0.7, c='red', label='Hα')
plt.legend()
plt.show()
```

- Uses `plt.scatter` under the hood; all extra `**kwargs` are passed
  through to matplotlib.
- Axis labels are auto-populated with units when the column is a
  `Quantity` (e.g. `"ra [deg]"`).
- Internally calls `self.to_numpy()` (the **class method**, which takes
  only `self`) to extract the `dunits` dict for labelling.

---

## Design Decisions & Patterns

| Decision | Rationale |
|---|---|
| **`__dict__`-based storage** (no `__slots__`) | Columns are dynamic; users can add / remove at runtime. KISS. |
| **Not a `dict` subclass** | `dict` cannot natively hold row-oriented multi-column semantics. A class with dunder overloads gives cleaner syntax (`obj[0:3]`, `obj['x']`). |
| **`@constant` returns a `property`** | Standard Python idiom for class-level constants. |
| **Module-level functions** (`add_dict`, `stack_dict`, `compare_dict`, `to_numpy`, `merge_dict`, …) | Operate on plain dicts; reusable outside the class. Class methods are thin wrappers. |
| **`compare_dict` before any mutation** | DRY: `__eq__`, `__ne__`, `compare`, and `__add__` all delegate to one function. |
| **`@ensure_consistency` / `@auto_pad_consistency`** | Decorator pattern to enforce the invariant *"all columns same length"* without repeating checks in every method. |
| **`SizeError` exception** | Explicit, catchable error for inconsistent-length operations. |
| **`deepcopy` for `copy()`** | Guarantees independence of nested mutable objects. |
| **`logging` for warnings, `print` for user info** | `logging` is configurable; `print` is for interactive / `verbose=True` output. |

### SOLID / KISS notes

- **Single Responsibility** – each module function does one thing.
- **Open / Closed** – extend by subclassing; no need to modify `classUtils`.
- **KISS** – the "object *is* a dict" mental model keeps the API small.
- **DRY** – `__eq__` / `__ne__` / `compare` / `__add__` all call `compare_dict`;
  `from_file` / `from_fits` / `from_table` / `from_dict` form a thin delegation chain.

---

## Known Limitations

1. **`to_pandas()` loses units.** Pandas has no built-in unit support.
   Use `to_table()` → `QTable` if units matter.
2. **Ragged columns** (different lengths per column) are tolerated but
   most operations (`sort`, `drop`, `to_table`, `save`) will fail or
   silently truncate. Call `check_consistency()` first.
3. **`__len__`** returns the **max** column length when lengths disagree.
   Use `_nelements()` for the full per-column picture.
4. **`add_dict`** returns `{}` on mismatch rather than raising.
   Check the return type or use `stack=True`.
5. **`save()` / `from_fits()`** contain a legacy `hourAngle`
   special-case. Remove in your subclass if unneeded.
6. **`from_csv`** is fragile with mixed-type columns and embedded
   commas. Prefer FITS / ECSV for production data.
7. **`stack_dict`** transpose logic (`zip(*l1)`) assumes a rectangular
   structure; deeply nested or ragged data will misbehave.
8. **No `__iter__` / `__next__`** – the class docstring mentions iteration
   but it is not implemented. Use `for i in range(len(obj)): obj[i]`.
9. **`__repr__`** returns `str(self.__dict__)` with no class name,
   making it hard to distinguish subclasses in the REPL.
10. **`merge_dict`** mutates its input dicts in-place
    (`d1[k1] = [d1[k1]]`) when a scalar value is found.
    Make defensive copies before calling.

---

## Bug Fixes Applied (v1.4)

The following bugs were identified and fixed in this revision:

| # | Location | Bug | Fix |
|---|----------|-----|-----|
| 1 | `classUtils.__neq__` | Method named `__neq__` but Python dispatches `!=` to `__ne__`. The `!=` operator only works by accident (fallback to `not __eq__`). | Renamed to `__ne__`. |
| 2 | `add_dict` – unit mismatch log | Logs `b1.unit` twice instead of `b1.unit` / `b2.unit`. | Changed second occurrence to `b2.unit`. |
| 3 | `merge_dict` – `same_units` init | `{'k': True for k in keys}` creates a single key `'k'` instead of one entry per key. | Changed to `{k: True for k in keys}`. |
| 4 | `merge_dict` – `same_type` check | Compares whole dicts (`dtype1 == dtype2`) inside the per-key loop instead of comparing individual keys. | Changed to `dtype1.get(k,'') == dtype2.get(k,'')`. |
| 5 | `classUtils.pivot` | Returns `classUtils(**new_dict)` (base class) instead of `self.__class__(**new_dict)`, losing the subclass. | Changed to `self.__class__(**new_dict)`. |
| 6 | `classUtils.plot` | Calls `self.to_numpy(self.__dict__)`. The class method `to_numpy(self)` takes **zero** additional arguments → `TypeError: to_numpy() takes 1 positional argument but 2 were given`. | Changed to `self.to_numpy()`. |
| 7 | `classUtils.from_csv` | File opened with `open()` but never closed on the success path. `NoneExist` variable may be unbound if the multi-row `try` block raises before assignment. | Use `with open(...) as file:` context manager. Pre-initialise `NoneExist = False`. |
| 8 | `classUtils.__len__` | Delegates to `_nelements()` which can return a `list` when column lengths disagree. `__len__` must return a non-negative `int`. | Return `max(n)` when `n` is a list. |
| 9 | `classUtils.drop` | Iterates `ind` in the order given. Unsorted input produces wrong slices. | Sort indices: `ind = np.sort(np.array(iarray))`. |
| 10 | `classUtils.__getitem__` (str) | Uses `getattr(self, item)` which raises `AttributeError`, not `KeyError`. | Use `self.__dict__[item]`. |

---

## Additional Improvements Recommended

These are **not yet applied** but are strongly recommended:

| # | Recommendation |
|---|---------------|
| 1 | Add `from __future__ import annotations` and type hints to all public signatures. |
| 2 | Implement `__iter__` / `__next__` so the iterator claim in the class docstring is true. |
| 3 | Add a `_frozen` flag checked in `__setitem__`, `remove_column`, etc. for read-only mode. |
| 4 | Remove the module-level `warnings.filterwarnings("error", …)` side-effect; move into a setup function. |
| 5 | Remove the unused `import astropy` (only sub-modules are used). |
| 6 | Remove the hard-coded `hourAngle` special-case in `save()` / `from_fits()`. |
| 7 | Replace ad-hoc `print(...)` calls in `merge_dict`, `drop`, `to_table` with `logging.warning` / `logging.info`. |
| 8 | Make `from_dict` and `merge_dict` use defensive copies of their input dicts. |
| 9 | Add `TypeError` catch in `compare_dict` (`order=False` path) for non-sortable values. |
| 10 | Add a unit test suite (pytest) beyond doctests for edge cases: empty objects, single-row objects, all-None columns, mixed-unit stacking. |
| 11 | Move `SizeError` to the top of the module (before its first use in docstrings). |
| 12 | Fix `__repr__` to include the class name: `f"{self.__class__.__name__}({self.__dict__})"`. |
| 13 | Remove the unreachable `continue` after `return {}` in `add_dict`. |
| 14 | Fix docstring typos: "quantiy" → "quantity", "withe" → "with", "then end" → "the end", "Trye" → "Try", "Paraameters" → "Parameters", "dicitonary" → "dictionary", "objecy" → "object". |
| 15 | Remove the phantom `stack` parameter from `stack_dict` docstring. |

---

## Testing

The module ships with **doctests** in every function and method docstring.

### Run doctests

```bash
cd ScientificContainerClass
python -m doctest classUtils.py -v
```

or from within Python:

```python
import doctest
import classUtils
doctest.testmod(classUtils, verbose=True, optionflags=doctest.ELLIPSIS)
```

The module's `__main__` block already runs doctests:

```python
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)
```

### pytest wrapper

```python
# test_classUtils.py
import doctest
import classUtils

def test_doctests():
    results = doctest.testmod(classUtils, optionflags=doctest.ELLIPSIS)
    assert results.failed == 0, f"{results.failed} doctest(s) failed"
```

```bash
pytest test_classUtils.py -v
```

### Recommended additional test cases

- Empty object: `MyClass()` → `len == 0`, `keys() == []`
- Single-row object: `MyClass(x=1, y='a')` → `len == 1`
- All-scalar object → `to_table()` should work or raise gracefully
- Mixed-unit `add_dict` → returns `{}`
- `from_csv` round-trip with `None` values
- `drop` with unsorted indices
- `pivot` on a subclass → returns the subclass, not `classUtils`
- `plot` with a non-Quantity column → no unit in label
- `ensure_consistency` decorator → raises `ValueError` on mismatch
- `auto_pad_consistency` → correct padding value (`NaN` vs `None`)

---

## Changelog

| Version | Date | Notes |
|---|---|---|
| 1.0 | 2023-06 | Initial release |
| 1.1 | 2024-01 | Added `pivot`, `from_csv`, `to_csv`, `to_hdu` |
| 1.2 | 2024-09 | `@ensure_consistency`, `@auto_pad_consistency`, `filter`, `plot` |
| 1.3 | 2025-03 | `stack_dict`, `use_list`, `auto_pad_consistency` refinements |
| **1.4** | **2026-10** | **Bug fixes:** `__ne__` rename, `add_dict` log typo, `merge_dict` unit / type checks, `pivot` class bug, `plot` `to_numpy` call, `from_csv` file leak, `__len__` protocol fix, `drop` index sorting, `__getitem__` KeyError fix. **Docs:** exhaustive README. |

---

## Contributing / Extending

### Minimal subclass

```python
from classUtils import classUtils

class MyData(classUtils):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # any post-processing, validation, defaults …

    # add domain-specific methods here
    def compute_peak_flux(self):
        self['peak_flux'] = self['flux'].max()
        return self
```

### Guidelines

- Follow the existing **numpydoc** docstring style.
- Add **doctests** to every new public method.
- Run `python -m doctest classUtils.py` before committing.
- Do **not** add hard-coded column-name special-cases to the base class;
  handle those in the concrete subclass.
- Use `logging.warning` / `logging.error` for error paths;
  `print` only for `verbose=True` user-facing output.
- Keep the "object is a dict" invariant: every column is accessible
  via both `obj['key']` and `obj.key`.

---

## License

This project is licensed under the **GNU General Public License v3.0**.

```
Copyright (C) 2023-2026  Wing-Fai Thi

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
```
```


"""Small compatibility shim for environments without dm-tree.

This project only needs ``tree.map_structure`` in the inference path, so we
provide a minimal recursive implementation that handles the common Python
container types used by GR00T inputs.
"""

from __future__ import annotations

from collections.abc import Mapping


def _is_namedtuple_instance(value) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields")


def map_structure(func, *structures):
    if not structures:
        raise TypeError("map_structure requires at least one structure")

    first = structures[0]

    if isinstance(first, Mapping):
        keys = list(first.keys())
        for other in structures[1:]:
            if not isinstance(other, Mapping) or list(other.keys()) != keys:
                raise TypeError("All mapping structures must have identical keys and order")
        return first.__class__(
            (key, map_structure(func, *(structure[key] for structure in structures)))
            for key in keys
        )

    if _is_namedtuple_instance(first):
        for other in structures[1:]:
            if not _is_namedtuple_instance(other) or type(other) is not type(first):
                raise TypeError("All namedtuple structures must have the same type")
        return type(first)(
            *(map_structure(func, *(structure[idx] for structure in structures)) for idx in range(len(first)))
        )

    if isinstance(first, tuple):
        for other in structures[1:]:
            if not isinstance(other, tuple) or len(other) != len(first):
                raise TypeError("All tuple structures must have identical length")
        return tuple(
            map_structure(func, *(structure[idx] for structure in structures))
            for idx in range(len(first))
        )

    if isinstance(first, list):
        for other in structures[1:]:
            if not isinstance(other, list) or len(other) != len(first):
                raise TypeError("All list structures must have identical length")
        return [
            map_structure(func, *(structure[idx] for structure in structures))
            for idx in range(len(first))
        ]

    return func(*structures)

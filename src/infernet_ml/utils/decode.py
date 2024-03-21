"""
Helper functions for decoding bytes stored in eth-abi format
"""
from typing import Any

import eth_abi
import numpy as np


def convert_double(num: int) -> float:
    """Converts SD59x18 to float

    Args:
        num (int): SD59x18

    Returns:
        float: parsed float
    """
    is_negative: bool = num < 0
    neg_str: str = "-" if is_negative else ""
    num_str: str = str(abs(num))

    # Prepare returned string
    returned: str = neg_str

    # If len < 18, prepend zeroes
    if len(num_str) < 18:
        prepend_zeroes: str = "0" * (18 - len(num_str))
        returned += f"0.{prepend_zeroes}{num_str}"
    else:
        # Get decimals
        decimal_index = len(num_str) - 18
        decimals: str = num_str[decimal_index:]

        # If length exactly 18 decimals
        if decimal_index == 0:
            returned += f"0.{decimals}"
        # If leading prefix
        else:
            leading: str = num_str[0:decimal_index]
            returned += f"{leading}.{decimals}"

    return float(returned)


def decode_multidim_array(
    arr: bytes, solidity_type_str: str, convert_to_float: bool = True
) -> Any:
    """decodes a multidim array

    Args:
        arr (bytes): byte that represents multi dim array
        solidity_type_str (str): corresponding solidity type string, e.g. int256[][]
        convert_to_float (bool, optional): whether or not to convert
        the resulting array value to floats. Defaults to True.

    Returns:
        Any: a list (potentially nested) of values
    """
    decoded_tuple: tuple[Any] = eth_abi.decode([solidity_type_str], arr)  # type: ignore
    np_arr = np.array(decoded_tuple[0])
    if convert_to_float:
        vectorized = np.vectorize(convert_double)
        np_arr = vectorized(np_arr)

    return np_arr.tolist()


def decode_vector(vector: list[int]) -> list[float]:
    """Parses list of SD59x18 ints to floats

    Args:
        vector (list[int]): list of SD59x18 ints

    Returns:
        list[float]: parsed floats
    """
    converted: list[float] = []
    for vec in vector:
        converted.append(convert_double(vec))
    return converted

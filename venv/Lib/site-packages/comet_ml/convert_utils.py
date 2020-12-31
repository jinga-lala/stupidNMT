# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.ml
#  Copyright (C) 2015-2020 Comet ML INC
#  This file can not be copied and/or distributed without the express
#  permission of Comet ML Inc.
# *******************************************************

import inspect
import logging

from ._typing import Any, Optional, Type

LOGGER = logging.getLogger(__name__)


def convert_tensor_to_numpy(tensor):
    """
    Convert from various forms of pytorch tensors
    to numpy arrays.

    Note: torch tensors can have both "detach" and "numpy"
    methods, but numpy() alone will fail if tensor.requires_grad
    is True.
    """
    if hasattr(tensor, "detach"):  # pytorch tensor with attached gradient
        tensor = tensor.detach()

    if hasattr(tensor, "numpy"):  # pytorch tensor
        tensor = tensor.numpy()

    return tensor


def convert_to_scalar(user_data, dtype=None):
    # type: (Any, Optional[Type]) -> Any
    """
    Try to ensure that the given user_data is converted back to a
    Python scalar, and of proper type (if given).
    """
    # Fast-path for types and class, we currently does not apply any conversion
    # to classes
    if inspect.isclass(user_data):

        if dtype and not isinstance(user_data, dtype):
            raise TypeError("%r is not of type %r" % (user_data, dtype))

        return user_data

    # First try to convert tensorflow tensor to numpy objects
    try:
        if hasattr(user_data, "numpy"):
            user_data = user_data.numpy()
    except Exception:
        LOGGER.debug(
            "Failed to convert tensorflow tensor %r to numpy object",
            user_data,
            exc_info=True,
        )

    # Then try to convert numpy object to a Python scalar
    try:
        if hasattr(user_data, "item") and callable(user_data.item):
            user_data = user_data.item()
    except Exception:
        LOGGER.debug(
            "Failed to convert object %r to Python scalar", user_data, exc_info=True,
        )

    if dtype is not None and not isinstance(user_data, dtype):
        raise TypeError("%r is not of type %r" % (user_data, dtype))

    return user_data


def convert_to_list(items, dtype=None):
    """
    Take an unknown item and convert to a list of scalars
    and ensure type is dtype, if given.
    """
    # First, convert it to numpy if possible:
    if hasattr(items, "numpy"):  # pytorch tensor
        items = convert_tensor_to_numpy(items)
    elif hasattr(items, "eval"):  # tensorflow tensor
        items = items.eval()

    # Next, handle numpy array:
    if hasattr(items, "tolist"):
        if len(items.shape) != 1:
            raise ValueError("list should be one dimensional")
        return items.tolist()
    else:
        # assume it is something with numbers in it:
        return [convert_to_scalar(item, dtype=dtype) for item in items]

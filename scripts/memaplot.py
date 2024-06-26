#! /usr/bin/env python3
import os

PRINT_DEBUG = False

FLUSH_INSTR_NONE = "none"


def print_debug(message):
    if PRINT_DEBUG:
        print(message)


def dir_path(path):
    """
    Checks if the given directory path is valid.

    :param path: directory path to the results folder
    :return: bool representing if path was valid
    """
    if os.path.isdir(path):
        return path
    else:
        raise ValueError("The path to the results directory is not valid.")


def valid_path(path):
    return path if os.path.isfile(path) else dir_path(path)


def values_as_string(values):
    return ", ".join(map(str, values))

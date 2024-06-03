#! /usr/bin/env python3

import os

PRINT_DEBUG = False

KEY_ACCESS_SIZE = "access_size"
KEY_BANDWIDTH_GiB = "bandwidth"
KEY_BANDWIDTH_GB = "bandwidth_gb"
KEY_BM_NAME = "bm_name"
KEY_BM_TYPE = "bm_type"
KEY_CHUNK_SIZE = "min_io_chunk_size"
KEY_CUSTOM_OPS = "custom_operations"
# KEY_EXPLODED_NUMA_MEMORY_NODES = "benchmarks.config.numa_memory_nodes"
# KEY_EXPLODED_NUMA_TASK_NODES = "benchmarks.config.numa_task_nodes"
KEY_LAT_AVG = "latency.avg"
KEY_LAT_STDDEV = "latency.std_dev"
KEY_MATRIX_ARGS = "matrix_args"
KEY_MEMORY_REGION_SIZE = "memory_region_size"
KEY_NUMA_TASK_NODES = "numa_task_nodes"
KEY_M0_NUMA_MEMORY_NODES = "m0_numa_nodes"
KEY_M1_NUMA_MEMORY_NODES = "m1_numa_nodes"
KEY_OPERATION = "operation"
KEY_OPERATION_COUNT = "number_operations"
KEY_OPS_PER_SECOND = "ops_per_second"
KEY_PARTITION_COUNT = "number_partitions"
KEY_RANDOM_DISTRIBUTION = "random_distribution"
KEY_RUN_TIME = "run_time"
KEY_SUB_BM_NAMES = "sub_bm_names"
KEY_TAG = "tag"
KEY_THREAD_COUNT = "number_threads"
KEY_THREADS = "threads"
KEY_THREADS_LEVELED = "benchmarks.results.threads"
KEY_FLUSH_INSTRUCTION = "flush_instruction"


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


def get_single_list_value(values):
    assert len(values) == 1
    return values[0]

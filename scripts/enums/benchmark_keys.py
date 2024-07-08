from enum import StrEnum


class BMKeys(StrEnum):
    ACCESSED_BYTES = "accessed_bytes"
    TOTAL_ACCESSES = "total_accesses"
    ACCESS_SIZE = "access_size"
    BANDWIDTH_GiB = "bandwidth"
    BANDWIDTH_GB = "bandwidth_gb"
    BM_GROUP = "bm_name"
    BM_NAME = "bm_name"
    BM_SUB_NAMES = "sub_bm_names"
    BM_TYPE = "bm_type"
    CHUNK_SIZE = "min_io_chunk_size"
    CUSTOM_OPS = "custom_operations"
    EXEC_MODE = "exec_mode"
    EXPLODED_NUMA_MEMORY_NODES = "benchmarks.config.numa_memory_nodes"  # legacy
    EXPLODED_NUMA_MEMORY_NODES_M0 = "benchmarks.config.m0_numa_nodes"
    EXPLODED_NUMA_MEMORY_NODES_M1 = "benchmarks.config.m1_numa_nodes"
    EXPLODED_NUMA_TASK_NODES = "benchmarks.config.numa_task_nodes"
    EXPLODED_THREAD_CORES = "benchmarks.config.thread_cores"
    LAT_AVG = "latency.avg"
    LAT_MEDIAN = "latency.median"
    LAT_STDDEV = "latency.std_dev"
    MATRIX_ARGS = "matrix_args"
    MEMORY_REGION_SIZE = "memory_region_size"
    NUMA_TASK_NODES = "numa_task_nodes"
    NUMA_MEMORY_NODES = "numa_memory_nodes"  # legacy
    NUMA_MEMORY_NODES_M0 = "m0_numa_nodes"
    NUMA_MEMORY_NODES_M1 = "m1_numa_nodes"
    OPERATION = "operation"
    OPERATION_COUNT = "number_operations"
    PARTITION_COUNT = "number_partitions"
    RANDOM_DISTRIBUTION = "random_distribution"
    RUN_TIME = "run_time"
    TAG = "tag"
    THREAD_COUNT = "number_threads"
    THREADS = "threads"
    THREADS_LEVELED = "benchmarks.results.threads"
    FLUSH_INSTRUCTION = "flush_instruction"
    EXEC_TIME = "execution_time"
    LABEL = "label"
    AVG_ACCESS_LATENCY = "avg_access_latency"

import glob
import json
import os
import sys

import pandas as pd

from enums.benchmark_keys import BMKeys
from enums.file_names import FILE_TAG_SUBSTRING
from memaplot import FLUSH_INSTR_NONE
from enums.benchmark_groups import BMGroups

PRINT_DEBUG = False


def print_debug(message):
    if PRINT_DEBUG:
        print(message)


def has_label(path, label):
    with open(path) as f:
        json_list = json.load(f)
        for json_obj in json_list:
            if "label" not in json_obj.keys() or json_obj["label"] != label:
                return False
        return True


def add_label(path, label):
    with open(path) as f:
        json_list = json.load(f)
        for json_object in json_list:
            json_object["label"] = label

    with open(path, "w") as f:
        json.dump(json_list, f)


def pretty_print(path):
    with open(path) as f:
        json_list = json.load(f)
        for count, json_obj in enumerate(json_list):
            print(json_obj.keys())
            print(json_obj["label"])
            print(json_obj["bm_name"])
            print(json_obj["bm_type"])


# Credits to Michele Piccolini: https://stackoverflow.com/a/61269285
def flatten_nested_json_df(df, deny_explosion_list):
    df = df.reset_index()

    print_debug(f"original shape: {df.shape}")
    print_debug(f"original columns: {df.columns}")
    print_debug(f"deny explosion for: {deny_explosion_list}")

    # search for columns to explode/flatten
    s = (df.map(type) == list).all()  # noqa: E721
    print_debug(s)
    list_columns = s[s].index.tolist()

    s = (df.map(type) == dict).all()  # noqa: E721
    dict_columns = s[s].index.tolist()

    print_debug(f"lists: {list_columns}, dicts: {dict_columns}")
    while len(list_columns) > 0 or len(dict_columns) > 0:
        new_columns = []

        for col in dict_columns:
            print_debug(f"flattening: {col}")
            # explode dictionaries horizontally, adding new columns
            horiz_exploded = pd.json_normalize(df[col]).add_prefix(f"{col}.")
            horiz_exploded.index = df.index
            df = pd.concat([df, horiz_exploded], axis=1).drop(columns=[col])
            new_columns.extend(horiz_exploded.columns)  # inplace

        for col in list_columns:
            if col in deny_explosion_list:
                print_debug(f"skip exploding: {col}")
                continue

            ends_with_deny_key = False
            for deny_key in deny_explosion_list:
                if col.endswith(deny_key):
                    ends_with_deny_key = True
                    break

            if ends_with_deny_key:
                print_debug(f"skip exploding: {col}")
                continue

            print_debug(f"exploding: {col}")
            # explode lists vertically, adding new columns
            df = df.drop(columns=[col]).join(df[col].explode().to_frame())
            new_columns.append(col)

        # check if there are still dict o list fields to flatten
        s = (df[new_columns].map(type) == list).all()  # noqa: E721
        list_columns = s[s].index.tolist()
        s = (df[new_columns].map(type) == dict).all()  # noqa: E721
        dict_columns = s[s].index.tolist()

        print_debug(f"lists: {list_columns}, dicts: {dict_columns}")

        # shorten column names
        def append_short_name(column, renamed_columns):
            split_keywords = ["benchmarks.results.", "benchmarks.config."]
            for split_keyword in split_keywords:
                if split_keyword in column:
                    renamed_columns.append(column.split(split_keyword)[-1:][0])
                    return True
            return False

        renamed_columns = []
        for column in df.columns:
            appended = append_short_name(column, renamed_columns)
            if not appended:
                renamed_columns.append(column)

        # check if name transformation created duplicates
        print_debug(f"original columns: {df.columns}")
        print_debug(f"renamed columns: {renamed_columns}")
        print_debug(f"set renamed columns: {set(renamed_columns)}")
        assert len(renamed_columns) == len(df.columns)
        assert len(renamed_columns) == len(set(renamed_columns))

        df.columns = renamed_columns

    print_debug(f"final shape: {df.shape}")
    print_debug(f"final columns: {df.columns}")
    return df


def parse_matrix_jsons(results, supported_bm_groups: list[BMGroups]):
    # collect jsons containing matrix arguments
    matrix_jsons = None
    if os.path.isfile(results):
        if not results.endswith(".json"):
            sys.exit("Result path is a single file but is not a .json file.")
        matrix_jsons = [results]
    else:
        matrix_jsons = [path for path in glob.glob(results + "/*.json")]

    # create json file list

    dfs = []
    for path in matrix_jsons:
        # Get the tag from the file name.
        tag = ""
        if FILE_TAG_SUBSTRING in path:
            path_parts = path.split(FILE_TAG_SUBSTRING)
            assert (
                len(path_parts) == 2
            ), "Make sure that the substring {} appears only once in a result file name.".format(FILE_TAG_SUBSTRING)
            tag_part = path_parts[-1]
            assert "-" not in tag_part, "Make sure that the tag is the last part of the name before the file extension."
            assert "_" not in tag_part, "Make sure that the tag is the last part of the name before the file extension."
            tag = tag_part.split(".")[0]

        df = pd.read_json(path)
        df[BMKeys.TAG] = tag
        dfs.append(df)

    df = pd.concat(dfs)
    bm_names = df[BMKeys.BM_GROUP].unique()
    print("Existing BM groups: {}".format(bm_names))

    print("Supported BM groups: {}".format([x.value for x in supported_bm_groups]))

    df = df[(df[BMKeys.BM_GROUP].isin(supported_bm_groups)) & (df[BMKeys.BM_TYPE] == "single")]
    df = flatten_nested_json_df(
        df,
        [
            BMKeys.MATRIX_ARGS,
            BMKeys.THREADS_LEVELED,
            BMKeys.EXPLODED_NUMA_MEMORY_NODES_M0,
            BMKeys.EXPLODED_NUMA_MEMORY_NODES_M1,
            BMKeys.EXPLODED_NUMA_TASK_NODES,
            BMKeys.EXPLODED_THREAD_CORES,
        ],
    )

    # If only latency benchnarks have been performed, the dataframe does not have a KEY_ACCESS_SIZE column so it
    # must be added.
    if BMKeys.ACCESS_SIZE not in df.columns:
        df[BMKeys.ACCESS_SIZE] = -1
    df[BMKeys.ACCESS_SIZE] = df[BMKeys.ACCESS_SIZE].fillna(-1)
    df[BMKeys.ACCESS_SIZE] = df[BMKeys.ACCESS_SIZE].astype(int)

    # For read benchmarks, an additional flush instruction will never be performed. As 'none' is also one of the
    # valid flush instructions, we set the corresponding value to 'none'. If only read benchnarks have been
    # performed, the dataframe does note have a KEY_FLUSH_INSTRUCTION column so it must be added.
    if BMKeys.FLUSH_INSTRUCTION not in df.columns:
        df[BMKeys.FLUSH_INSTRUCTION] = FLUSH_INSTR_NONE
    df[BMKeys.FLUSH_INSTRUCTION] = df[BMKeys.FLUSH_INSTRUCTION].fillna(FLUSH_INSTR_NONE)
    if BMKeys.BANDWIDTH_GiB in df.columns:
        df[BMKeys.BANDWIDTH_GB] = df[BMKeys.BANDWIDTH_GiB] * (1024**3 / 1e9)

    df = stringify_nodes(df)
    return df


def stringify_nodes(df):
    df[BMKeys.NUMA_MEMORY_NODES_M0] = df[BMKeys.NUMA_MEMORY_NODES_M0].transform(lambda x: ",".join(str(i) for i in x))
    df[BMKeys.NUMA_MEMORY_NODES_M1] = df[BMKeys.NUMA_MEMORY_NODES_M1].transform(lambda x: ",".join(str(i) for i in x))
    df[BMKeys.NUMA_TASK_NODES] = df[BMKeys.NUMA_TASK_NODES].transform(lambda x: ",".join(str(i) for i in x))
    return df

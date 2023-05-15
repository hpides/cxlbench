import json
import pandas as pd


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

    print(f"original shape: {df.shape}")
    print(f"original columns: {df.columns}")

    # search for columns to explode/flatten
    s = (df.applymap(type) == list).all()
    list_columns = s[s].index.tolist()

    s = (df.applymap(type) == dict).all()
    dict_columns = s[s].index.tolist()

    print(f"lists: {list_columns}, dicts: {dict_columns}")
    while len(list_columns) > 0 or len(dict_columns) > 0:
        new_columns = []

        for col in dict_columns:
            print(f"flattening: {col}")
            # explode dictionaries horizontally, adding new columns
            horiz_exploded = pd.json_normalize(df[col]).add_prefix(f"{col}.")
            horiz_exploded.index = df.index
            df = pd.concat([df, horiz_exploded], axis=1).drop(columns=[col])
            new_columns.extend(horiz_exploded.columns)  # inplace

        for col in list_columns:
            if col in deny_explosion_list:
                print(f"skip exploding: {col}")
                continue
            print(f"exploding: {col}")
            # explode lists vertically, adding new columns
            df = df.drop(columns=[col]).join(df[col].explode().to_frame())
            new_columns.append(col)

        # check if there are still dict o list fields to flatten
        s = (df[new_columns].applymap(type) == list).all()
        list_columns = s[s].index.tolist()

        s = (df[new_columns].applymap(type) == dict).all()
        dict_columns = s[s].index.tolist()

        print(f"lists: {list_columns}, dicts: {dict_columns}")

        # shorten column names
        renamed_columns = []
        for column in df.columns:
            if "latency." in column:
                if "results.latency." in column:
                    renamed_columns.append(column.split("results.")[-1:][0])
                    continue

                renamed_columns.append(column)
            else:
                renamed_columns.append(column.split(".")[-1:][0])

        # check if name transformation created duplicates
        print(renamed_columns)
        assert len(renamed_columns) == len(df.columns)
        assert len(renamed_columns) == len(set(renamed_columns))

        df.columns = renamed_columns

    print(f"final shape: {df.shape}")
    print(f"final columns: {df.columns}")
    return df

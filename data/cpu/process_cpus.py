#! /usr/bin/env python3

import argparse
import csv
import os
import pandas as pd

def parse_csv(file_path):
    column_names = []
    columns = []

    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        for row in reader:
            # Skip rows with only one value
            if len(row) <= 1:
                continue

            # First value is the column name, remaining are column values
            column_names.append(row[0])
            columns.append(row[1:])
    column_names[0] = "Name"
    df = pd.DataFrame(columns).T
    df.columns = column_names
    return df

if __name__ == "__main__":
    # parse args + check for correctness and completeness of args
    parser = argparse.ArgumentParser()

    parser.add_argument("csv_path", help="path to the results directory")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path.startswith("./") and not csv_path.startswith("/"):
        csv_path = "./" + csv_path

    if os.path.isfile(csv_path):
        parts = csv_path.rsplit("/", 1)
        assert len(parts)
        output_dir_string = parts[0]
        output_dir_string = output_dir_string
    else:
        assert os.path.isdir(csv_path)
        output_dir_string = csv_path
    output_dir = os.path.abspath(output_dir_string)
    os.makedirs(output_dir, exist_ok=True)

if os.path.isfile(csv_path):
    if not csv_path.endswith(".csv"):
        sys.exit("Input file path is a single file but is not a .csv file.")
    file_paths = [csv_path]
else:
    file_paths = [path for path in glob.glob(csv_path + "/*.csv")]

for path in file_paths:
    df  = parse_csv(path)
    # df = df[df["Name"].str.contains("Platinum|Gold")]
    df = df[df["Name"].str.contains("Platinum", case=False, na=False)]
    # df = df[(df["Intel® In-memory Analytics Accelerator (IAA)"].str.strip() == "") | (df["Intel® In-memory Analytics Accelerator (IAA)"].str.contains("0"))]
    df = df[df["Memory Types"].str.contains("2DPC") | df["Memory Types"].str.contains("2 DPC")]
    # selection = [
    #     "Name",
    #     "Recommended Customer Price",
    #     "Scalability",
    #     "Cache",
    #     "Max Memory Size (dependent on memory type)",
    #     # "Memory Types",
    #     "Max # of Memory Channels",
    #     "Max # of PCI Express Lanes",
    #     "PCI Express Revision"
    # ]
    # print(df[selection])
    prefix = path.rsplit("/",1)[-1].split("_",1)[0]
    df.to_csv(f"{output_dir}/{prefix}-platinum-2dpc.csv")

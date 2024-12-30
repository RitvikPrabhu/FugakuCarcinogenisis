#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import pandas as pd


def get_data(gene_sample_file, normal_gene_list_file):
    gene_sample_df = pd.read_csv(
        gene_sample_file,
        sep=r"\s+",
        header=None,
        names=["row", "column", "mutation", "gene", "sample"],
        skiprows=1,
    )
    gene_sample_pivot = gene_sample_df.pivot(
        index="gene", columns="sample", values="mutation"
    ).fillna(0)
    gene_sample_pivot = gene_sample_pivot.apply(
        lambda x: x.map(lambda y: 1 if y > 0 else 0)
    )

    normal_gene_list_df = pd.read_csv(
        normal_gene_list_file,
        sep=r"\s+",
        header=None,
        names=["gene", "sample"],
    )
    normal_samples = normal_gene_list_df["sample"].unique()
    normal_columns = [f"normal_{sample}" for sample in normal_samples]
    normal_data = pd.DataFrame(
        0,
        index=gene_sample_pivot.index.union(normal_gene_list_df["gene"].unique()),
        columns=normal_columns,
    )

    for _, row in normal_gene_list_df.iterrows():
        gene = row["gene"]
        sample = f'normal_{row["sample"]}'
        normal_data.at[gene, sample] = 1

    gene_sample_pivot = (
        pd.concat([gene_sample_pivot, normal_data], axis=1).fillna(0).astype(int)
    )
    return gene_sample_pivot, normal_data


def sort_data(gene_sample_pivot):
    # Identify columns that start with 'TCGA'
    tcga_columns = [col for col in gene_sample_pivot.columns if col.startswith("TCGA")]

    if not tcga_columns:
        print("No TCGA columns found. Skipping sorting step.")
        return gene_sample_pivot

    # Count the number of 1's in TCGA columns for each gene
    gene_sample_pivot["tcga_count"] = gene_sample_pivot[tcga_columns].sum(axis=1)

    # Sort the DataFrame based on 'tcga_count'
    gene_sample_pivot = gene_sample_pivot.sort_values(by="tcga_count", ascending=True)

    # Drop the 'tcga_count' column
    return gene_sample_pivot.drop(columns=["tcga_count"])


def process_gene_data(gene_sample_file, normal_gene_list_file, output_file):
    gene_sample_pivot, normal_data = get_data(gene_sample_file, normal_gene_list_file)
    gene_sample_pivot = sort_data(gene_sample_pivot)

    num_rows, num_cols = gene_sample_pivot.shape
    num_normal_samples = normal_data.shape[1]
    num_tumor_samples = num_cols - num_normal_samples

    print("Number of rows (genes):", num_rows)
    print("Number of columns (samples):", num_cols)

    # Write the file
    try:
        with open(output_file, "w") as f:
            # Write the header line
            header_line = (
                f"{num_rows} {num_cols} -1 {num_tumor_samples} {num_normal_samples}\n"
            )
            f.write(header_line)

            for row_tuple in gene_sample_pivot.itertuples(index=False):
                row_values = map(str, row_tuple)
                f.write("".join(row_values) + "\n")

        print(f"File successfully written to {output_file}")

    except Exception as e:
        print(f"Error writing file: {e}")

    try:
        # Gene -> row index
        gene_map_file = str(output_file) + ".gene_map"
        with open(gene_map_file, "w") as f_gene_map:
            for i, gene in enumerate(gene_sample_pivot.index):
                f_gene_map.write(f"{gene}\t{i}\n")

        # Sample -> column index
        # col_map_file = str(output_file) + ".col_map"
        # with open(col_map_file, "w") as f_col_map:
        #    for j, sample in enumerate(gene_sample_pivot.columns):
        #        f_col_map.write(f"{sample}\t{j}\n")

        print(f"Row (gene) mapping written to {gene_map_file}")
        # print(f"Column (sample) mapping written to {col_map_file}")

    except Exception as e:
        print(f"Error writing mapping files: {e}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("gene_sample_file", type=Path)
    parser.add_argument("normal_gene_list_file", type=Path)
    parser.add_argument("output_file", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python process_gene_data.py <gene_sample_file> <normal_gene_list_file> <output_file>"
        )
        sys.exit(1)
    args = get_args()
    if not args.gene_sample_file.exists():
        print(f"Gene sample file not found: {args.gene_sample_file}")
        sys.exit(1)
    if not args.normal_gene_list_file.exists():
        print(f"Normal gene list file not found: {args.normal_gene_list_file}")
        sys.exit(1)
    process_gene_data(
        args.gene_sample_file, args.normal_gene_list_file, args.output_file
    )

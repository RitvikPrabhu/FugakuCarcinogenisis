#!/usr/bin/env python
import argparse
import sys
import re
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
    gene_sample_pivot = gene_sample_pivot.apply(lambda x: x.map(lambda y: 1 if y > 0 else 0))

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

    gene_sample_pivot = pd.concat([gene_sample_pivot, normal_data], axis=1).fillna(0).astype(int)
    return gene_sample_pivot, normal_data


def sort_data(gene_sample_pivot):
    tcga_columns = [col for col in gene_sample_pivot.columns if col.startswith("TCGA")]

    if not tcga_columns:
        print("No TCGA columns found. Skipping sorting step.")
        return gene_sample_pivot

    gene_sample_pivot["tcga_count"] = gene_sample_pivot[tcga_columns].sum(axis=1)
    gene_sample_pivot = gene_sample_pivot.sort_values(by="tcga_count", ascending=True)
    return gene_sample_pivot.drop(columns=["tcga_count"])


def perform_coverage_analysis(gene_sample_pivot, solution_file):
    num_rows, num_cols = gene_sample_pivot.shape
    print("Number of rows (genes):", num_rows)
    print("Number of columns (samples):", num_cols)

    tcga_columns = set(col for col in gene_sample_pivot.columns if col.startswith("TCGA"))
    normal_columns = set(col for col in gene_sample_pivot.columns if col.startswith("normal"))

    unique_values_tumor = set()
    unique_values_normal = set()
    line_count = 0

    pattern = re.compile(r"\((.*?)\)")

    with open(solution_file, "r") as file:
        for line in file:
            match = pattern.search(line)
            if not match:
                print(f"Could not extract genes from line: {line.strip()}")
                continue

            genes_str = match.group(1)
            row_names = [gene.strip() for gene in genes_str.split(",")]
            try:
                filtered_df = gene_sample_pivot.loc[row_names]
            except KeyError:
                print(f"Error: One or more genes in {row_names} not found in the data.")
                continue

            intersection = set(filtered_df.columns[(filtered_df > 0).all()])
            unique_values_tumor.update(intersection)
            unique_values_normal.update(intersection)

            line_count += 1
            tumor_coverage = len(unique_values_tumor & tcga_columns)
            normal_coverage = len(unique_values_normal & normal_columns)
            print(f"Line {line_count}: Tumor coverage: {tumor_coverage} / {len(tcga_columns)}, "
                  f"Normal coverage: {normal_coverage} / {len(normal_columns)}")

            if tcga_columns.issubset(unique_values_tumor):
                print(f"All TCGA columns are covered at line {line_count}.")
                break

    if tcga_columns.issubset(unique_values_tumor):
        print("All TCGA columns are covered.")
    else:
        print(f"Some TCGA columns are missing. Covered: {len(unique_values_tumor & tcga_columns)} / {len(tcga_columns)}")

    total_normal = len(normal_columns)
    if normal_columns.issubset(unique_values_normal):
        print("All normal columns are covered.")
    else:
        covered = len(unique_values_normal & normal_columns)
        print(f"Some normal columns are missing. Covered: {covered} / {total_normal} "
              f"({(covered / total_normal) * 100:.2f}%)")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("gene_sample_file", type=Path, help="Path to gene sample file")
    parser.add_argument("normal_gene_list_file", type=Path, help="Path to normal gene list file")
    parser.add_argument("solution_file", type=Path, help="Path to solution file (e.g., solution_4hit.txt)")
    return parser.parse_args()


def main():
    args = get_args()

    for file in [args.gene_sample_file, args.normal_gene_list_file, args.solution_file]:
        if not file.exists():
            print(f"File not found: {file}")
            sys.exit(1)

    gene_sample_pivot, _ = get_data(args.gene_sample_file, args.normal_gene_list_file)
    gene_sample_pivot = sort_data(gene_sample_pivot)
    perform_coverage_analysis(gene_sample_pivot, args.solution_file)


if __name__ == "__main__":
    main()


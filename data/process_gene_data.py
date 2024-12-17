import os
import sys

import pandas as pd


def process_gene_data(gene_sample_file, normal_gene_list_file, output_file):
    gene_sample_df = pd.read_csv(
        gene_sample_file,
        sep="\s+",
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

    num_tumor_samples = gene_sample_pivot.shape[1]

    normal_gene_list_df = pd.read_csv(
        normal_gene_list_file, sep="\s+", header=None, names=["gene", "sample"]
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
    num_cancer_samples = normal_data.shape[1]

    gene_sample_pivot = (
        pd.concat([gene_sample_pivot, normal_data], axis=1).fillna(0).astype(int)
    )

    # ---- Sorting Step Starts Here ----

    # Identify columns that start with 'TCGA'
    tcga_columns = [col for col in gene_sample_pivot.columns if col.startswith("TCGA")]

    if tcga_columns:
        # Count the number of 1's in TCGA columns for each gene
        gene_sample_pivot["tcga_count"] = gene_sample_pivot[tcga_columns].sum(axis=1)

        # Sort the DataFrame based on 'tcga_count'
        gene_sample_pivot = gene_sample_pivot.sort_values(
            by="tcga_count", ascending=True
        )

        # Drop the 'tcga_count' column
        gene_sample_pivot = gene_sample_pivot.drop(columns=["tcga_count"])
    else:
        print("No TCGA columns found. Skipping sorting step.")

    # ---- Sorting Step Ends Here ----

    num_rows, num_cols = gene_sample_pivot.shape
    print("Number of rows (genes):", num_rows)
    print("Number of columns (samples):", num_cols)

    result = gene_sample_pivot.reset_index().melt(
        id_vars="gene", var_name="sample", value_name="mutation"
    )

    result = result.sort_values(["gene", "sample"]).reset_index(drop=True)
    result["row"] = result.index // len(result["sample"].unique())
    result["column"] = result.index % len(result["sample"].unique())
    final_result = result[["row", "column", "mutation", "gene", "sample"]]

    try:
        with open(output_file, "w") as f:
            f.write(
                f"{num_rows} {num_cols} -1 {num_tumor_samples} {num_cancer_samples}\n"
            )
            final_result.to_csv(f, sep=" ", index=False, header=False)
        print(f"File successfully written to {output_file}")
    except Exception as e:
        print(f"Error writing file: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python process_gene_data.py <gene_sample_file> <normal_gene_list_file> <output_file>"
        )
        sys.exit(1)
    gene_sample_file = sys.argv[1]
    normal_gene_list_file = sys.argv[2]
    output_file = sys.argv[3]
    if not os.path.exists(gene_sample_file):
        print(f"Gene sample file not found: {gene_sample_file}")
        sys.exit(1)
    if not os.path.exists(normal_gene_list_file):
        print(f"Normal gene list file not found: {normal_gene_list_file}")
        sys.exit(1)
    process_gene_data(gene_sample_file, normal_gene_list_file, output_file)

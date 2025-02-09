#!/usr/bin/env python
import argparse
import sys
from pathlib import Path


def load_gene_map(gene_map_file: Path) -> dict:
    gene_map = {}
    with gene_map_file.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    index = int(parts[1])
                    gene_map[index] = parts[0]
                except ValueError:
                    continue
    return gene_map


def replace_indices_with_gene_names(solution_file: Path, gene_map: dict) -> None:
    lines = solution_file.read_text().splitlines(keepends=True)
    updated_lines = []
    for line in lines:
        start = line.find("(")
        end = line.find(")")
        if start != -1 and end != -1 and end > start:
            indices_str = line[start + 1 : end]
            try:
                indices = [
                    int(token.strip())
                    for token in indices_str.split(",")
                    if token.strip()
                ]
            except ValueError:
                updated_lines.append(line)
                continue
            gene_names = [gene_map.get(idx, str(idx)) for idx in indices]
            gene_names_str = ", ".join(gene_names)
            new_line = line[: start + 1] + gene_names_str + line[end:]
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)
    solution_file.write_text("".join(updated_lines))
    print(f"Solution file '{solution_file}' has been updated.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Replace indices in a solution file with gene names using a gene map file."
    )
    parser.add_argument("solution_file", type=Path, help="Path to the solution file.")
    parser.add_argument("gene_map_file", type=Path, help="Path to the gene map file.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.solution_file.exists():
        print(f"Solution file not found: {args.solution_file}")
        sys.exit(1)
    if not args.gene_map_file.exists():
        print(f"Gene map file not found: {args.gene_map_file}")
        sys.exit(1)
    gene_map = load_gene_map(args.gene_map_file)
    replace_indices_with_gene_names(args.solution_file, gene_map)


if __name__ == "__main__":
    main()

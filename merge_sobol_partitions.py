"""Merge results from partitioned Sobol sequence runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd


def find_partition_dirs(base_dir: Path, algorithm: str, env_name: str, seed: int) -> List[Path]:
    """Find all partition directories for a given experiment.
    
    Parameters
    ----------
    base_dir : Path
        Base directory containing results.
    algorithm : str
        Algorithm name (e.g., 'ppo').
    env_name : str
        Environment name (e.g., 'cc_acrobot').
    seed : int
        Seed used for the experiment.
        
    Returns
    -------
    List[Path]
        List of partition directories, sorted by partition id.
    """
    # Try multiple possible patterns for directory structure
    # Pattern 1: results/sobol/{algorithm}_{env_name}_{seed}/partition_{id_sub}_of_{n_sub}
    experiment_dir1 = base_dir / f"{algorithm}_{env_name}_{seed}"
    
    # Pattern 2: results/sobol/{algorithm}_{env_name}/{seed}/partition_{id_sub}_of_{n_sub}
    experiment_dir2 = base_dir / f"{algorithm}_{env_name}" / str(seed)
    
    # Pattern 3: results/sobol/{algorithm}/{algorithm}_{env_name}/{seed}/partition_{id_sub}_of_{n_sub}
    experiment_dir3 = base_dir / algorithm / f"{algorithm}_{env_name}" / str(seed)
    
    # Try each pattern
    for experiment_dir in [experiment_dir1, experiment_dir2, experiment_dir3]:
        if experiment_dir.exists():
            # Find all partition directories
            partition_dirs = []
            for path in experiment_dir.iterdir():
                if path.is_dir() and path.name.startswith("partition_"):
                    partition_dirs.append(path)
            
            if partition_dirs:
                # Sort by partition id
                def get_partition_id(path: Path) -> int:
                    # Extract id from "partition_{id}_of_{n}"
                    name = path.name
                    id_str = name.split("_")[1]
                    return int(id_str)
                
                partition_dirs.sort(key=get_partition_id)
                return partition_dirs
    
    # If no pattern worked, raise an error
    raise ValueError(
        f"No partition directories found. Tried:\n"
        f"  - {experiment_dir1}\n"
        f"  - {experiment_dir2}\n"
        f"  - {experiment_dir3}"
    )


def validate_partitions(partition_dirs: List[Path]) -> Dict[str, any]:
    """Validate that partitions are consistent.
    
    Parameters
    ----------
    partition_dirs : List[Path]
        List of partition directories.
        
    Returns
    -------
    Dict[str, any]
        Partition metadata.
    """
    metadata_list = []
    
    for partition_dir in partition_dirs:
        # Look for partition metadata file
        metadata_files = list(partition_dir.glob("**/sobol_partition_*.json"))
        
        if metadata_files:
            with open(metadata_files[0], "r") as f:
                metadata = json.load(f)
                metadata_list.append(metadata)
    
    if not metadata_list:
        print("Warning: No partition metadata files found. Proceeding anyway.")
        return {"seed": None, "n_sub": len(partition_dirs)}
    
    # Verify all partitions have the same seed and n_sub
    seed = metadata_list[0]["seed"]
    n_sub = metadata_list[0]["n_sub"]
    
    for i, metadata in enumerate(metadata_list):
        if metadata["seed"] != seed:
            raise ValueError(
                f"Partition {i} has different seed: {metadata['seed']} vs {seed}"
            )
        if metadata["n_sub"] != n_sub:
            raise ValueError(
                f"Partition {i} has different n_sub: {metadata['n_sub']} vs {n_sub}"
            )
        if metadata["id_sub"] != i:
            print(
                f"Warning: Partition {i} has id_sub={metadata['id_sub']} "
                f"(expected {i}). Results may not merge correctly."
            )
    
    print(f"✓ Validated {len(metadata_list)} partitions (seed={seed}, n_sub={n_sub})")
    return {"seed": seed, "n_sub": n_sub}


def merge_results(partition_dirs: List[Path], output_dir: Path) -> None:
    """Merge results from all partitions.
    
    Parameters
    ----------
    partition_dirs : List[Path]
        List of partition directories, sorted by partition id.
    output_dir : Path
        Directory to save merged results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files across partitions
    all_csv_files = {}
    for partition_dir in partition_dirs:
        csv_files = list(partition_dir.glob("**/*.csv"))
        for csv_file in csv_files:
            # Get relative path from partition directory
            rel_path = csv_file.relative_to(partition_dir)
            
            # Skip job-specific directories (e.g., "0/", "1/", etc.)
            if rel_path.parts[0].isdigit():
                # This is a job directory, get the file within it
                if str(rel_path) not in all_csv_files:
                    all_csv_files[str(rel_path)] = []
                all_csv_files[str(rel_path)].append(csv_file)
    
    if not all_csv_files:
        print("Warning: No CSV files found to merge.")
        return
    
    # Merge each set of CSV files
    for rel_path, csv_files in all_csv_files.items():
        print(f"\nMerging {len(csv_files)} files for {rel_path}")
        
        # Read all CSV files
        dfs = []
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {csv_file}: {e}")
        
        if not dfs:
            continue
        
        # Concatenate
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Save merged file
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_file, index=False)
        print(f"  Saved merged file: {output_file} ({len(merged_df)} rows)")
    
    # Copy other files (e.g., configs, logs)
    print("\nCopying additional files from first partition...")
    first_partition = partition_dirs[0]
    for file_path in first_partition.glob("**/*"):
        if file_path.is_file() and not file_path.suffix == ".csv":
            rel_path = file_path.relative_to(first_partition)
            output_file = output_dir / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, output_file)
    
    print(f"\n✓ Merged results saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Merge results from partitioned Sobol sequence runs."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/sobol"),
        help="Base directory containing results (default: results/sobol)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        help="Algorithm name (e.g., 'ppo')",
    )
    parser.add_argument(
        "--environment",
        type=str,
        required=True,
        help="Environment name (e.g., 'cc_acrobot')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed used for the experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for merged results (default: {base_dir}/{algorithm}_{environment}/{seed}/merged)",
    )
    
    args = parser.parse_args()
    
    # Find partition directories
    print(f"Searching for partitions in {args.base_dir}...")
    partition_dirs = find_partition_dirs(args.base_dir, args.algorithm, args.environment, args.seed)
    print(f"Found {len(partition_dirs)} partitions:")
    for partition_dir in partition_dirs:
        print(f"  - {partition_dir.name}")
    
    # Validate partitions
    print("\nValidating partitions...")
    metadata = validate_partitions(partition_dirs)
    
    # Determine output directory
    if args.output_dir is None:
        # Use the same parent directory as the partitions
        partition_parent = partition_dirs[0].parent
        output_dir = partition_parent / "merged"
    else:
        output_dir = args.output_dir
    
    # Merge results
    print("\nMerging results...")
    merge_results(partition_dirs, output_dir)
    
    # Save merge metadata
    merge_metadata = {
        "algorithm": args.algorithm,
        "environment": args.environment,
        "seed": args.seed,
        "n_partitions": len(partition_dirs),
        "sobol_seed": metadata.get("seed"),
        "n_sub": metadata.get("n_sub"),
    }
    
    with open(output_dir / "merge_metadata.json", "w") as f:
        json.dump(merge_metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("Merge complete!")
    print("="*80)


if __name__ == "__main__":
    main()

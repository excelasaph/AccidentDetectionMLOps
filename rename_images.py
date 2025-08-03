#!/usr/bin/env python3
"""
Image Renaming Script for Accident Detection Dataset

This script renames all images in the dataset to follow a consistent naming convention:
Format: {folder_name}_{split}_{sequence_number}_{frame_number}.jpg

Example transformations:
- test1_18.jpg in test/Accident/ → accident_test_1_1.jpg
- test1_19.jpg in test/Accident/ → accident_test_1_2.jpg
- test2_5.jpg in train/Non Accident/ → non_accident_train_1_1.jpg

Author: Generated for AccidentDetectionMLOps project
Date: August 2, 2025
"""

import os
import shutil
from collections import defaultdict
import argparse


def extract_sequence_id(filename):
    """Extract sequence ID from filename."""
    if '_' in filename:
        # Standard pattern: test10_33.jpg -> test10
        return '_'.join(filename.split('_')[:-1])
    else:
        # Handle files like acc1.jpg -> acc1
        return filename.split('.')[0]


def sort_files_in_sequence(files):
    """Sort files within a sequence by frame number."""
    def extract_frame_number(filepath):
        filename = os.path.basename(filepath)
        try:
            # Try standard pattern: test10_33.jpg -> 33
            if '_' in filename:
                return int(filename.split('_')[-1].split('.')[0])
            else:
                # Handle alternative patterns
                if '(' in filename and ')' in filename:
                    # Pattern like acc1 (2).jpg -> 2
                    return int(filename.split('(')[-1].split(')')[0])
                else:
                    # Fallback to alphabetical
                    return filename
        except (ValueError, IndexError):
            # Fallback to alphabetical sorting
            return filename
    
    return sorted(files, key=extract_frame_number)


def rename_images_in_directory(data_dir, dry_run=False):
    """
    Rename all images in the dataset to follow consistent naming convention.
    
    Args:
        data_dir (str): Path to data directory containing train/test/val folders
        dry_run (bool): If True, only print what would be renamed without actually renaming
    
    Returns:
        dict: Summary of renaming operations
    """
    
    # Define splits and their corresponding folders
    splits = ['train', 'test', 'val']
    label_folders = ['Accident', 'Non Accident']
    
    summary = {
        'total_files': 0,
        'renamed_files': 0,
        'errors': [],
        'operations': []
    }
    
    print(f"{'='*60}")
    print(f"IMAGE RENAMING SCRIPT")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Dry run mode: {dry_run}")
    print(f"{'='*60}")
    
    for split in splits:
        split_path = os.path.join(data_dir, split)
        
        if not os.path.exists(split_path):
            print(f"⚠️  Warning: {split_path} does not exist, skipping...")
            continue
        
        print(f"\n📁 Processing {split.upper()} split:")
        print(f"{'─'*40}")
        
        for label_folder in label_folders:
            folder_path = os.path.join(split_path, label_folder)
            
            if not os.path.exists(folder_path):
                print(f"⚠️  Warning: {folder_path} does not exist, skipping...")
                continue
            
            # Create folder name for renaming (lowercase, replace spaces with underscores)
            folder_name = label_folder.lower().replace(' ', '_')
            
            print(f"\n  📂 Processing {label_folder} folder:")
            
            # Get all image files
            image_files = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    image_files.append(os.path.join(folder_path, file))
            
            summary['total_files'] += len(image_files)
            
            if not image_files:
                print(f"    ℹ️  No image files found")
                continue
            
            print(f"    📊 Found {len(image_files)} image files")
            
            # Group images by sequence
            sequences = defaultdict(list)
            for file_path in image_files:
                filename = os.path.basename(file_path)
                sequence_id = extract_sequence_id(filename)
                sequences[sequence_id].append(file_path)
            
            print(f"    🔗 Grouped into {len(sequences)} sequences")
            
            # Process each sequence
            sequence_counter = 1
            for sequence_id, sequence_files in sequences.items():
                # Sort files in sequence by frame number
                sorted_files = sort_files_in_sequence(sequence_files)
                
                print(f"      🎬 Sequence {sequence_counter} (original: {sequence_id}): {len(sorted_files)} frames")
                
                # Rename each file in the sequence
                for frame_number, old_file_path in enumerate(sorted_files, 1):
                    old_filename = os.path.basename(old_file_path)
                    file_extension = os.path.splitext(old_filename)[1]
                    
                    # Create new filename: {folder_name}_{split}_{sequence_number}_{frame_number}.jpg
                    new_filename = f"{folder_name}_{split}_{sequence_counter}_{frame_number}{file_extension}"
                    new_file_path = os.path.join(folder_path, new_filename)
                    
                    operation = {
                        'old_path': old_file_path,
                        'new_path': new_file_path,
                        'old_name': old_filename,
                        'new_name': new_filename,
                        'split': split,
                        'label': label_folder,
                        'sequence': sequence_counter,
                        'frame': frame_number
                    }
                    
                    summary['operations'].append(operation)
                    
                    if old_file_path != new_file_path:  # Only rename if names are different
                        try:
                            if not dry_run:
                                # Check if target file already exists
                                if os.path.exists(new_file_path):
                                    error_msg = f"Target file already exists: {new_filename}"
                                    summary['errors'].append(error_msg)
                                    print(f"        ❌ Error: {error_msg}")
                                    continue
                                
                                # Perform the rename
                                os.rename(old_file_path, new_file_path)
                                summary['renamed_files'] += 1
                                print(f"        ✅ {old_filename} → {new_filename}")
                            else:
                                print(f"        🔄 Would rename: {old_filename} → {new_filename}")
                                summary['renamed_files'] += 1
                        
                        except Exception as e:
                            error_msg = f"Error renaming {old_filename}: {str(e)}"
                            summary['errors'].append(error_msg)
                            print(f"        ❌ {error_msg}")
                    else:
                        print(f"        ⏭️  Skipped (already correct): {old_filename}")
                
                sequence_counter += 1
    
    return summary


def print_summary(summary, dry_run=False):
    """Print a summary of the renaming operations."""
    print(f"\n{'='*60}")
    print(f"RENAMING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {summary['total_files']}")
    print(f"Files {'would be ' if dry_run else ''}renamed: {summary['renamed_files']}")
    print(f"Errors encountered: {len(summary['errors'])}")
    
    if summary['errors']:
        print(f"\n❌ ERRORS:")
        for error in summary['errors']:
            print(f"  • {error}")
    
    # Show some example operations
    if summary['operations']:
        print(f"\n📋 EXAMPLE OPERATIONS:")
        examples = summary['operations'][:5]  # Show first 5
        for op in examples:
            action = "Would rename" if dry_run else "Renamed"
            print(f"  {action}: {op['old_name']} → {op['new_name']}")
        
        if len(summary['operations']) > 5:
            print(f"  ... and {len(summary['operations']) - 5} more operations")
    
    print(f"{'='*60}")


def main():
    """Main function to handle command line arguments and execute renaming."""
    parser = argparse.ArgumentParser(
        description="Rename images in accident detection dataset to follow consistent naming convention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rename_images.py --data-dir ./data --dry-run
  python rename_images.py --data-dir ./data --execute
  
Naming Convention:
  {folder_name}_{split}_{sequence_number}_{frame_number}.jpg
  
  Where:
  - folder_name: 'accident' or 'non_accident'
  - split: 'train', 'test', or 'val'
  - sequence_number: Sequential number starting from 1 for each split/label combination
  - frame_number: Sequential frame number starting from 1 for each sequence
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Path to data directory containing train/test/val folders (default: ./data)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be renamed without actually renaming files'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually perform the renaming operations'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dry_run and not args.execute:
        print("❌ Error: You must specify either --dry-run or --execute")
        parser.print_help()
        return
    
    if args.dry_run and args.execute:
        print("❌ Error: Cannot specify both --dry-run and --execute")
        parser.print_help()
        return
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory does not exist: {args.data_dir}")
        return
    
    # Ask for confirmation if executing
    if args.execute:
        print("⚠️  WARNING: This will permanently rename files in your dataset!")
        confirmation = input("Are you sure you want to continue? (yes/no): ")
        if confirmation.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return
    
    # Perform the renaming
    summary = rename_images_in_directory(args.data_dir, dry_run=args.dry_run)
    
    # Print summary
    print_summary(summary, dry_run=args.dry_run)
    
    if args.dry_run:
        print("\n💡 To actually perform the renaming, run with --execute flag")


if __name__ == "__main__":
    main()

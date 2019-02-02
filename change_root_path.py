"""
On windows I sort images by folders (one folder - one class).
On linux all images is in one folder. Folds csv file is creating
on windows and this module is changing root path for fold.csv
"""
import os
import argparse
import pandas as pd


def change_root(df_path, new_root):
    df = pd.read_csv(df_path)
    df['path'] = df['path'].map(lambda x: os.path.join(new_root, os.path.basename(x)))
    df.to_csv(df_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str)
    parser.add_argument('--root', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.root = '/mnt/hdd2/datasets/naive_data/shot_dataset/shot_total_bigger/all_images/'
    args.csv = '/mnt/hdd2/datasets/naive_data/shot_dataset/shot_total_bigger/fixed_folds.csv'
    change_root(args.csv, args.root)

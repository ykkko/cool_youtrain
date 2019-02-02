import os
import glob
import argparse
import numpy as np
import pandas as pd


def make_fold_csv(dataset_root, dst_path):
    """
    Make fold file for dir with class's subfolders:

    train_set
    |-- class_1
    |-- class_2
    `-- class_3

    :param dataset_root:
    :param dst_path:
    :return:
    """

    filenames = glob.glob(os.path.join(dataset_root, '**/*.*'))
    val_part = 0.2

    df = pd.DataFrame(filenames, columns=['path'])
    df['path'] = df['path'].map(lambda x: x.replace('\\', '/'))
    df['fname'] = df['path'].map(lambda x: os.path.basename(x))
    np.random.shuffle(df.values)
    df['label'] = df['path'].map(lambda x: x.split('/')[-2])

    label_counts = df.groupby('label').size().to_dict()
    print(label_counts)
    label, count = list({k: v for k, v in label_counts.items() if v == min(label_counts.values())}.items())[0]
    print(label, count)

    df['fold'] = 0
    val_count = int(val_part * count)
    print(val_count)
    for label in df['label'].unique():
        train_count = len(df[df['label'] == label]) - val_count
        df.loc[df['label'] == label, 'fold'] = [0] * val_count + [1] * train_count
    df.to_csv(dst_path)


def change_path(df_path, new_path, dst_path=None):
    """

    :param df_path:
    :param path2change:
    :return:
    """
    df = pd.read_csv(df_path)
    df['path'] = df['path'].map(lambda x: os.path.join(new_path, os.path.basename(x)))
    df.to_csv(dst_path if dst_path else df_path)


def check_value_counts(path):
    check_folds = pd.read_csv(path)
    for label in check_folds['label'].unique():
        print(label, '\n', check_folds[check_folds['label'] == label]['fold'].value_counts())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--fold_path', type=str)
    # parser.add_argument('--')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # args.root = '/mnt/hdd2/datasets/naive_data/shot_dataset/shot_total_bigger/fixed_train'
    # args.fold_path = '/mnt/hdd2/datasets/naive_data/shot_dataset/shot_total_bigger/folds_fixed_train.csv'
    make_fold_csv(args.root, args.fold_path)
    check_value_counts(args.fold_path)

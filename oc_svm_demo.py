import argparse

from loglizer.loglizer.models import OneClassSVM
from loglizer.loglizer import dataloader, preprocessing
from utils import Timed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='HDFS', choices=["HDFS", "BGL"], type=str,
                        help='Target dataset. Default HDFS')
    args = parser.parse_args()

    with Timed("Load data"):
        struct_log = f'data/{args.dataset}.seqs.csv'  # The structured log file
        if args.dataset == 'HDFS':
            (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(
                struct_log, window='session', train_ratio=0.5, split_type='uniform'
            )
        elif args.dataset == 'BGL':
            (x_train, y_train), (x_test, y_test) = dataloader.load_BGL(
                struct_log, train_ratio=0.5, split_type='uniform'
            )
        else:
            raise ValueError('Dataset is not supported')

    feature_extractor = preprocessing.FeatureExtractor()
    with Timed("Fit feature extractor and transform train"):
        x_train = feature_extractor.fit_transform(x_train, term_weighting='zero-mean')
    with Timed("Transform test"):
        x_test = feature_extractor.transform(x_test)

    model = OneClassSVM(max_iter=200)
    with Timed("Fit model"):
        model.fit(x_train[y_train == 0, :])  # Use only normal samples for training

    with Timed("Train evaluation"):
        model.evaluate(x_train, y_train)

    with Timed("Test evaluation"):
        precision, recall, f1 = model.evaluate(x_test, y_test)

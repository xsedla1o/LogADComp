import argparse

from loglizer.loglizer import dataloader, preprocessing
from loglizer.loglizer.models import LogClustering
from utils import Timed

max_dist = 0.3  # the threshold to stop the clustering process
anomaly_threshold = 0.3  # the threshold for anomaly detection

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
        x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    with Timed("Transform test"):
        x_test = feature_extractor.transform(x_test)

    model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
    with Timed("Fit model"):
        model.fit(x_train[y_train == 0, :])  # Use only normal samples for training

    with Timed("Train validation"):
        model.evaluate(x_train, y_train)

    with Timed("Test validation"):
        model.evaluate(x_test, y_test)

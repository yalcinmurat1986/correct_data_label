import os
import pandas
import logging
from argparse import ArgumentParser

from correct_data_labels_full import CorrectLabels

logging.basicConfig(
            format = '%(asctime)s:%(funcName)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from pathlib import Path
home = str(Path.home())

datasets = ['iris', 'mnist']
def start_correct(num_of_wrongs, 
                    repeats, 
                    split_rate,
                    split_rate_threshold,
                    data_path,
                    epochs,
                    steps,
                    decay_step,
                    dl,
                    save_path,
                    save_file_name,
                    dataset,
                    threshold,
                    min_num_predictions,
                    label_name):
    if not num_of_wrongs:
        num_of_wrongs = [25]
    else:
        num_of_wrongs = [int(v) for v in num_of_wrongs]
    if not repeats:
        repeats = [500]
    else:
        repeats = [int(v) for v in repeats]
    if not split_rate:
        split_rate = [0.1]
    else:
        split_rate = [float(v) for v in split_rate]

    def load_iris_dataset():
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        dataset = pandas.read_csv(url, names=names)
        return dataset
    
    if not data_path:
        data_path = os.path.join(home, 'hs/projects/CorrectDataLabel/data/train.csv')

    def load_mnist_dataset(data_path):
        t = pandas.read_csv(data_path)
        cols = list(t.columns)
        cols = cols[1:] + [cols[0]]
        dataset = t[cols]
        return dataset
    num_of_wrongs = num_of_wrongs
    repeats = repeats
    split_rate = split_rate
    results = []
    logger.info('loading dataset...')
    if dataset == 'iris':
        label_name = 'class'
        dataset = load_iris_dataset()
        logger.info('iris dataset loaded successfully...')
        logger.info(f'length of dataset : {len(dataset)}')
    elif dataset == 'mnist':
        dataset = load_mnist_dataset(data_path)
        logger.info('mnist dataset loaded successfully...')
        logger.info(f'length of dataset : {len(dataset)}')
    logger.info('experiment started...')
    for i in num_of_wrongs:
        for j in repeats:
            for k in split_rate:
                cl = CorrectLabels(dataset = dataset,
                                    label_column_name = label_name, # class
                                    epochs = epochs,
                                    steps = steps,
                                    decay_step = decay_step,
                                    num_of_wrongs = i, 
                                    repeats = j, 
                                    split_rate = k,
                                    split_rate_threshold = split_rate_threshold,
                                   mnist = True,
                                   threshold=threshold,
                                   min_num_predictions = min_num_predictions)
                logger.info(f'combination : {(i, j, k)}')
                if dl:
                    result = cl.correct_wrong_labels_cnn()
                else:
                    result = cl.correct_wrong_labels()
                results.extend(result)
    res = pandas.DataFrame(results)
    save_path = os.path.join(home, 'hs/projects/CorrectDataLabel/data/results')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f'{save_file_name}.csv' )
    res.to_csv(save_file)

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_of_wrongs', required = False,
    default = None, nargs = '+')
    parser.add_argument('--repeats', required = False, 
    default = None,  nargs = '+')
    parser.add_argument('--split_rate',  required = False, 
    default = None, nargs = '+')
    parser.add_argument('--split_rate_threshold',  required = False, 
    default = 0.6)
    parser.add_argument('--data_path', type = str, required = False)
    parser.add_argument('--epochs', type = int, required = False, default = 10)
    parser.add_argument('--steps', type = int, required = False, default = 5)
    parser.add_argument('--decay_step', type = int, required = False, default = 3)
    parser.add_argument('--dl', action = 'store_true')
    parser.add_argument('--save_path', type = str, required = False)
    parser.add_argument('--save_file_name', type = str, required = False, default = 'results')
    parser.add_argument('--dataset', type = str, required = False, default = 'mnist', choices = datasets)
    parser.add_argument('--threshold', type = float, required = False, default = 0.95)
    parser.add_argument('--min_num_predictions', type = int, required = False, default = 10)
    parser.add_argument('--label_name', type = str, default = 'label')
    args = parser.parse_args()

    start_correct(args.num_of_wrongs, 
                    args.repeats, 
                    args.split_rate,
                    args.split_rate_threshold,
                    args.data_path,
                    args.epochs,
                    args.steps,
                    args.decay_step,
                    args.dl,
                    args.save_path,
                    args.save_file_name,
                    args.dataset,
                    args.threshold,
                    args.min_num_predictions,
                    args.label_name)

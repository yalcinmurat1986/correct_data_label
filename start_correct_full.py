import pandas
import logging
from argparse import ArgumentParser

from correct_data_labels_full import CorrectLabels

logging.basicConfig(
            format = '%(asctime)s:%(funcName)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

datasets = ['iris', 'mnist']
def start_correct(num_of_wrongs, 
                    repeats, 
                    split_rate,
                    path,
                    epochs,
                    steps,
                    dl,
                    save_file_name,
                    dataset,
                    threshold,
                    min_num_predictions,
                    label_name):
    if not num_of_wrongs:
        num_of_wrongs = [4000, 5000, 7000]
    else:
        num_of_wrongs = [int(v) for v in num_of_wrongs]
    if not repeats:
        repeats = [1000]
    else:
        repeats = [int(v) for v in repeats]
    if not split_rate:
        split_rate = [0.1, 0.075, 0.05]
    else:
        split_rate = [float(v) for v in split_rate]
    if not path:
        path = '/home/dreamventures/hs/projects/CorrectDataLabel/data/train.csv'
        # path = '/Users/muratyalcin/Downloads/train.csv'

    def load_iris_dataset():
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        dataset = pandas.read_csv(url, names=names)
        return dataset
    def load_mnist_dataset(path):
        t = pandas.read_csv(path)
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
        dataset = load_iris_dataset()
        logger.info('iris dataset loaded successfully...')
        logger.info(f'length of dataset : {len(dataset)}')
    elif dataset == 'mnist':
        dataset = load_mnist_dataset(path)
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
                                    num_of_wrongs = i, 
                                    repeats = j, 
                                    split_rate = k,
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
    res.to_csv(f'{save_file_name}.csv')

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_of_wrongs', required = False,
    default = None, nargs = '+')
    parser.add_argument('--repeats', required = False, 
    default = None,  nargs = '+')
    parser.add_argument('--split_rate',  required = False, 
    default = None, nargs = '+')
    parser.add_argument('--path', type = str, required = False)
    parser.add_argument('--epochs', type = int, required = False, default = 10)
    parser.add_argument('--steps', type = int, required = False, default = 5)
    parser.add_argument('--dl', action = 'store_true')
    parser.add_argument('--save_file_name', type = str, required = False, default = 'results')
    parser.add_argument('--dataset', type = str, required = False, default = 'mnist', choices = datasets)
    parser.add_argument('--threshold', type = float, required = False, default = 0.90)
    parser.add_argument('--min_num_predictions', type = int, required = False, default = 3)
    parser.add_argument('--label_name', type = str, default = 'label')
    args = parser.parse_args()

    start_correct(args.num_of_wrongs, 
                    args.repeats, 
                    args.split_rate,
                    args.path,
                    args.epochs,
                    args.steps,
                    args.dl,
                    args.save_file_name,
                    args.dataset,
                    args.threshold,
                    args.min_num_predictions,
                    args.label_name)

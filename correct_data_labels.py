import pandas, random
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,\
AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from collections import defaultdict, Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from create_models import create_cnn_model, \
    create_cnn_model_2, create_baseline_model

class CorrectLabels:
    
    def __init__(self,
                 dataset,
                 label_column_name, 
                 num_of_wrongs, 
                 repeats, 
                 split_rate,
                 epochs,
                 iris = None,
                 mnist = None):
        self.epochs = epochs
        self.dataset = dataset
        self.label_column_name = label_column_name
        self.split_rate = split_rate 
        self.num_of_wrongs = num_of_wrongs
        self.repeats = repeats
        self.num_of_features = self.dataset.shape[1]-1
        self.labels = list(self.dataset[label_column_name].unique())
        self.num_of_labels = len(self.labels)
        self.mlmodels = self.form_ml_models()
        self.dlmodels = self.form_dl_models()
        #if iris:
        #    assert mnist is None
        #    self.dataset = self.load_iris_dataset()
        #if mnist:
        #    self.dataset = self.load_mnist_dataset()
        
    
    def load_iris_dataset(self):
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        dataset = pandas.read_csv(url, names=names)
        return dataset
    
    def load_mnist_dataset(self):
        t = pandas.read_csv('/Users/muratyalcin/Downloads/train.csv')
        cols = list(t.columns)
        cols = cols[1:] + [cols[0]]
        dataset = t[cols]
        return dataset
    
    def shuffle_dataset(self, dataset):
        shuffled_dataset = shuffle(dataset)
        #shuffled_dataset = shuffled_dataset.reset_index(drop=True)
        return shuffled_dataset
        
    def make_wrong(self, dataset):
        change_indexes = random.sample(range(0,len(dataset)+1),
                                       self.num_of_wrongs)
        trues = []
        wrongs = []
        wrong_dataset = dataset.copy()
        for i in change_indexes:
            true_label = self.dataset.at[i , self.label_column_name]
            trues.append(true_label)
            wrong_label = random.choice([i for i in self.labels if i != true_label])
            wrongs.append(wrong_label)
            wrong_dataset.at[i , self.label_column_name] = wrong_label
        return wrong_dataset, trues, wrongs, change_indexes
    
    def dataset_train_test_split(self, dataset):
        split_point = int(len(dataset)/self.split_rate)
        train_data = dataset[split_point:]
        test_data = dataset[:split_point]
        return train_data, test_data, split_point
    
    def df_to_vector(self, df):
        return df.values
        
    def x_y_split_vector(self, vector):
        X = vector[:,0:self.num_of_features]
        y = vector[:,self.num_of_features]
        return X, y 
        
    def form_ml_models(self):
        models = {}
        #models['LR'] = LogisticRegression(solver='liblinear', multi_class='ovr')
        models['KNN'] = KNeighborsClassifier()
        models['RF'] = RandomForestClassifier()
        #models['NB'] = GaussianNB()
        models['SVM'] = SVC(gamma='auto')
        #models['MultinomialNB'] = MultinomialNB()
        #models['AdaBoost'] = AdaBoostClassifier() 
        #models['GradientBoost'] = GradientBoostingClassifier()
        return models

    def form_dl_models(self):
        models = {}
        # models['baseline'] = create_baseline_model(self.num_of_labels, self.num_of_features)
        models['CNN'] = create_cnn_model(self.num_of_labels)
        models['CNN2'] = create_cnn_model_2(self.num_of_labels)
        return models       

    def fit_cnn(self, model, X_train, Y_train, X_val, Y_val):

        batch_size = 64
        # without data augmentation
        model.fit(X_train, Y_train, batch_size = batch_size, epochs = self.epochs, 
        validation_data = (X_val, Y_val), verbose = 2)

        return model     
    
    def fit_(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model
    
    def predict_(self, model,  X_test):
        predictions = model.predict(X_test)
        return predictions
        
    def multi_model_predict(self, X_train, y_train, X_test):
        preds = []
        for model_name, model in self.mlmodels.items():
            print(f'fitting and predicting with {model_name}')
            model = self.fit_(model, X_train, y_train)
            predictions = self.predict_(model, X_test)
            preds.append(predictions)
        return preds
    
    def multi_model_predict_cnn(self, X_train, Y_train, X_val, Y_val, X_test):
        preds = []
        for model_name, model in self.dlmodels.items():
            print(f'fitting and predicting with {model_name}')
            model = self.fit_cnn(model, X_train, Y_train, X_val, Y_val)
            predictions = self.predict_(model, X_test)
            # Convert one hot vectors to predictions classes 
            predictions = np.argmax(predictions,axis = 1) 
            preds.append(predictions)
        return preds
    
    
    def handle_tracker(self, tracker, wrong_dataset):
        wrong_data_labels = list(wrong_dataset[self.label_column_name])
        assert len(list(self.dataset[self.label_column_name])) == \
        len(list(wrong_dataset[self.label_column_name]))
        item_preds = defaultdict()
        model_guess = []
        for i in range(len(self.dataset)):
            if tracker[i]:
                item_preds[i] = max(Counter(tracker[i]), key=Counter(tracker[i]).get) 
                model_guess.append(max(Counter(tracker[i]), key=Counter(tracker[i]).get))
            else:
                item_preds[i] = wrong_data_labels[i]
                model_guess.append(wrong_data_labels[i])
        return item_preds, model_guess
   
    def compare(self, model_guess):
        actuals = list(self.dataset[self.label_column_name])
        predicted = model_guess 
        corrects = [i for i, j in enumerate(zip(actuals, model_guess)) if j[0] == j[1]]
        wrongs = [i for i, j in enumerate(zip(actuals, model_guess)) if j[0] != j[1]]
        return corrects, wrongs
 
    def evaluate(self, corrects, change_indexes, wrongs):
        return {
            'data length' : len(self.dataset),
            'split rate' : self.split_rate,
            'repeats' : self.repeats,
            'total wrongs start' : self.num_of_wrongs,
            'number of corrects' : len(corrects),
            'number of wrongs' : len(wrongs),
            'number of wrong indexes'  : len(change_indexes),
            'number of corrected' : len(set(change_indexes) & set(corrects)),
            'number of missed' : len(set(change_indexes) & set(wrongs)), 
            'number of wronged' : len((set(change_indexes) | set(wrongs)) - set(change_indexes))   
        }
        
    def correct_wrong_labels(self):
        tracker = defaultdict(list)
        wrong_dataset, trues, wrongs, change_indexes = self.make_wrong(self.dataset)
        for i in range(self.repeats):
            dataset = self.shuffle_dataset(wrong_dataset)
            train_data, test_data, split_point = self.dataset_train_test_split(wrong_dataset)
            train_data_ = self.df_to_vector(train_data)
            test_data_ = self.df_to_vector(test_data)
            X_train, y_train = self.x_y_split_vector(train_data_)
            X_test, y_test = self.x_y_split_vector(test_data_) 
            
            preds = self.multi_model_predict(X_train, y_train, X_test)
            num_models = len(self.mlmodels)
            assert len(preds[0]) == split_point
            y_indexes = list(test_data.index)
            for x in range(num_models):
                for i, index in enumerate(y_indexes):
                    tracker[index].append(preds[x][i])
        item_preds, model_guess = self.handle_tracker(tracker, wrong_dataset)
        corrects, wrongs = self.compare(model_guess)
        result = self.evaluate(corrects, change_indexes, wrongs)
        print('result : ', result) 
        return result
    
    
    def correct_wrong_labels_cnn(self):
        tracker = defaultdict(list)
        wrong_dataset, trues, wrongs, change_indexes = self.make_wrong(self.dataset)
        for i in range(self.repeats):
            print(f'processing {i}/{self.repeats}')
            dataset = self.shuffle_dataset(wrong_dataset)
            train_data, test_data, split_point = self.dataset_train_test_split(wrong_dataset)
            # Drop 'label' column
            X_train = train_data.drop(labels = ["label"],axis = 1) 
            Y_train = train_data["label"]

            X_test = test_data.drop(labels = ["label"],axis = 1) 
            Y_test = test_data["label"]
            # Normalize the data
            X_train = X_train / 255.0
            X_test = X_test / 255.0

            # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
            X_train = X_train.values.reshape(-1,28,28,1)
            X_test = X_test.values.reshape(-1,28,28,1)
            
            # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
            Y_train = to_categorical(Y_train, num_classes = 10)
            Y_test = to_categorical(Y_test, num_classes = 10)
            
            # Split the train and the validation set for the fitting
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1)

            
#             # Set a learning rate annealer
#             learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
#                                             patience=3, 
#                                             verbose=1, 
#                                             factor=0.5, 
#                                             min_lr=0.00001)

#             datagen = ImageDataGenerator(
#                     featurewise_center=False,  # set input mean to 0 over the dataset
#                     samplewise_center=False,  # set each sample mean to 0
#                     featurewise_std_normalization=False,  # divide inputs by std of the dataset
#                     samplewise_std_normalization=False,  # divide each input by its std
#                     zca_whitening=False,  # apply ZCA whitening
#                     rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#                     zoom_range = 0.1, # Randomly zoom image 
#                     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#                     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#                     horizontal_flip=False,  # randomly flip images
#                     vertical_flip=False)  # randomly flip images

#             datagen.fit(X_train)
            
#             # Fit the model
#             history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
#                               epochs = self.epochs, validation_data = (X_val,Y_val),
#                               verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
#                               , callbacks=[learning_rate_reduction])
            preds = self.multi_model_predict_cnn(X_train, Y_train, X_val, Y_val, X_test)
            num_models = len(self.dlmodels)
            assert len(preds[0]) == split_point
            y_indexes = list(test_data.index)
            for x in range(num_models):
                for i, index in enumerate(y_indexes):
                    tracker[index].append(preds[x][i])
        item_preds, model_guess = self.handle_tracker(tracker, wrong_dataset)
        corrects, wrongs = self.compare(model_guess)
        result = self.evaluate(corrects, change_indexes, wrongs)
        print('result : ', result) 
        return result
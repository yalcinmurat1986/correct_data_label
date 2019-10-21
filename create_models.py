from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.layers.normalization import BatchNormalization

         
def create_baseline_model(num_of_labels, num_of_features):
    # create model
    model = Sequential()
    model.add(Dense(num_of_features, input_dim=num_of_features, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_of_labels, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_cnn_model_2(num_of_labels):
    model = Sequential()

    model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',
                        input_shape = (28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
    model.add(BatchNormalization())
    #model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
    model.add(BatchNormalization())
    #model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                    optimizer = Adam(lr=1e-4), metrics=["accuracy"])
    return model


def create_cnn_model(num_of_labels):
    # Set the CNN model 
    # my CNN architechture is In -> 
    #[[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                        activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                        activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                        activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                        activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_labels, activation = "softmax"))
    
    # Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # Compile the model
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", 
                    metrics=["accuracy"])
    return model


def create_multi_cnn_model(num_of_labels):
    # BUILD CONVOLUTIONAL NEURAL NETWORKS
    nets = 15
    model = [0] *nets
    for j in range(nets):
        model[j] = Sequential()

        model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
        model[j].add(BatchNormalization())
        model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))
        model[j].add(BatchNormalization())
        model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
        model[j].add(BatchNormalization())
        model[j].add(Dropout(0.4))

        model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
        model[j].add(BatchNormalization())
        model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
        model[j].add(BatchNormalization())
        model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
        model[j].add(BatchNormalization())
        model[j].add(Dropout(0.4))
        model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
        model[j].add(BatchNormalization())
        model[j].add(Flatten())
        model[j].add(Dropout(0.4))
        model[j].add(Dense(num_of_labels, activation='softmax'))

        # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
        model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# def train_multi_cnn_model():
#     nets = 15
#     # DECREASE LEARNING RATE EACH EPOCH
#     annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
#     # TRAIN NETWORKS
#     history = [0] * nets
#     epochs = 45
#     for j in range(nets):
#         X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)
#         history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
#             epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  
#             validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)
#         print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
#             j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
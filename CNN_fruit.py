import keras

print('keras: ', keras.__version__)
from keras.models import Sequential
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import h5py

modelname = 'alexnet'
#modelname = 'lenet'

data_augmentation = True

seed = 10
# function for getting top-2 accuracy
def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

# if val_acc is > 0.9999, stop training
def check(epoch, logs):
    global model
    if logs.get('val_acc') > 0.9999:
        model.stop_training = True

# modelo 1: lenet
def lenet(m, n, nb_classes):
    model = Sequential()
    model.add(Convolution2D(6, 7, 7, border_mode='same', input_shape=(3, m, n), activation='relu', init='Orthogonal'))
    model.add(MaxPooling2D(data_format='channels_first', strides=2))

    model.add(Convolution2D(8, 5, 5, border_mode='same', activation='relu', init='Orthogonal'))
    model.add(MaxPooling2D(data_format='channels_first', strides=2))

    model.add(Flatten())
    model.add(Dense(60, activation='relu', init='glorot_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(45, activation='relu', init='glorot_normal'))
    model.add(Dropout(0.3))

    model.add(Dense(nb_classes, activation='softmax', init='glorot_normal'))
    for layer in model.layers:
        print (layer.output_shape)
    return model

# modelo 2: alexnet
def alexnet(m, n, nb_classes):
    model = Sequential()
    model.add(Convolution2D(32, 7, 7, border_mode='same', input_shape=(3, m, n), activation='relu', init='Orthogonal'))
    model.add(MaxPooling2D(data_format='channels_first', strides=2))

    model.add(Convolution2D(96, 5, 5, border_mode='same', activation='relu', init='Orthogonal'))
    model.add(MaxPooling2D(data_format='channels_first', strides=2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', init='Orthogonal'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', init='Orthogonal'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', init='Orthogonal'))
    model.add(MaxPooling2D(data_format='channels_first', strides=2))

    model.add(Flatten())
    model.add(Dense(192, activation='relu', init='glorot_normal'))
    model.add(Dropout(0.7))

    model.add(Dense(192, activation='relu', init='glorot_normal'))
    model.add(Dropout(0.7))

    model.add(Dense(nb_classes, activation='softmax', init='glorot_normal'))
    for layer in model.layers:
        print (layer.output_shape)
    return model

# load data
f = h5py.File('./fruits_downsampled_16.h5', 'r')
X = np.array(f.get('data'))
labels = np.array(f.get('label'))
f.close()

# print labels
nb_classes = max(labels) + 1
print ("Number of classes: " + str(nb_classes))

nsamples = X.shape[0]
m = X.shape[1]
n = X.shape[2]
print ('Image size: ' + str(m) + 'x' + str(n))

# rearrange to shape (sample, channel, m, n)
X = X.astype('float32')
X = np.swapaxes(np.swapaxes(X, 2, 3), 1, 2)
X = X.reshape(nsamples, 3, m, n)
X /= 255.0

# training parameters
batch_size = 16
nb_epoch = 300
learning_rate = 0.001

# training samples and repetitions
num_tr_samples = 8
rpt = 3

# initialize accuracies
accs = np.zeros(rpt)
top_2_accs = np.zeros(rpt)

# training loop
for k in range(rpt):

    # split training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, train_size=nb_classes*num_tr_samples, random_state=seed)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_train = np_utils.to_categorical(Y_train, nb_classes)

    # data augmentation
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 vertical_flip=True,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2)

    augmented = datagen.flow(X_train,Y_train,batch_size=batch_size)

    # call model
    if modelname == 'alexnet':
        model = alexnet(m, n, nb_classes)
    elif modelname == 'lenet':
        model = lenet(m, n, nb_classes)
    
    # choose optimizer
    optimizer = SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    # check callback: from check function
    check_cb = LambdaCallback(on_epoch_end=check)
    
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy',top_2_accuracy])

    # checkpoint
    filepath = './'+modelname + "/"+str(num_tr_samples)+"/fruit_weights_best_"+str(k)+".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    # fit and save in hist
    hist = model.fit_generator(augmented, steps_per_epoch=240, nb_epoch=nb_epoch, verbose=1, shuffle=False,
                     validation_data=(X_test, Y_test), callbacks=callbacks_list)

    #plt.figure("Accuracy")
    #plt.clf()
    #plt.plot(hist.history['acc'])
    #plt.plot(hist.history['val_acc'])
    #plt.figure("Loss")
    #plt.clf()
    #plt.plot(hist.history['loss'])
    #plt.plot(hist.history['val_loss'])
    #plt.pause(5)

    # get metrics from hist
    acc = max(hist.history['acc'])
    loss = min(hist.history['loss'])
    val_acc = max(hist.history['val_acc'])
    val_loss = min(hist.history['val_loss'])
    top_2 = max(hist.history['top_2_accuracy'])
    val_top_2 = max(hist.history['val_top_2_accuracy'])

    # print metrics
    print (str(k) + ' ' + str(loss) + ' ' + str(val_loss) + ' ' + str(acc) + ' ' + str(val_acc))
    print ('top-2 (train): '+str(top_2))
    print ('top-2 (val): '+ str(val_top_2))
    if val_acc == 1.0:
        break
    
    # save 
    f_handle = open(modelname+'_fruits_results_' + str(nsamples)+'.txt', 'a')
    f_handle.write(str(k) + ' ' + str(loss) + ' ' + str(val_loss) + ' ' + str(acc) + ' ' + str(val_acc) + '\n')
    f_handle.close()
    # if k == 0:
    #   plot(model, to_file='AlexNetmodel.png')
    
    # save accuracies 
    accs[k] = val_acc
    top_2_accs[k] = val_top_2

print("Saved model to disk")
print ('Average (top-1): ', np.average(accs))
print ('Average (top-2): ', np.average(top_2_accs))

f_handle = open(modelname+'_fruits_results_' + str(nsamples)+'.txt', 'a')
f_handle.write('Average (top-1): '+ str(np.average(accs))+'\n')
f_handle.write('Average (top-2): '+ str(np.average(top_2_accs))+'\n')
f_handle.close()

plt.show()
from __future__ import print_function

import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

dropOut = float(input('Dropout (a number between 0 & 1) >> '))
valData = input('Validation set? (y/n) >> ')
batchSize = input('Batch mode: (1.batch  2.mini-batch  3.stochastic) >> ')
epochs = int(input('Number of epochs >> '))

if batchSize == '1' or batchSize == 'batch':
    batch_size = 60000
elif batchSize == '2' or batchSize == 'mini-batch':
    batch_size = 128
elif batchSize == '3' or batchSize == 'stochastic':
    batch_size = 1
else:
    batch_size = None

num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Our neural-network is going to take a single vector for each training example,
# so we need to reshape the input so that each 28x28 image becomes a single 784 dimensional vector.
# We'll also scale the inputs to be in the range [0-1] rather than [0-255]
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if valData == 'y':
    valData = (x_test, y_test)
else:
    valData = None

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(dropOut))
model.add(Dense(512, activation='relu'))
model.add(Dropout(dropOut))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=epochs,
                    verbose=1,
                    validation_data=valData)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

loss = input('Do you want to plot loss values? (y/n)>> ')
if loss == 'y':
    trainCost = history.history['loss']
    testCost = history.history['val_loss']
    print(trainCost)
    print(testCost)
    hLine = []
    for i in range(1, epochs + 1):
        hLine.append(i)
    plt.plot(hLine, trainCost, 'r--', label="Train cost")
    plt.plot(hLine, testCost, 'g--', label="Test cost")
    plt.axis([1, epochs, 0, 0.3])
    plt.title('Loss plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()
    plt.close()

q = input('Do you want Precision/Recall/F1-score? (y/n)>> ')
if q == 'y':
    classLst = []
    for i in range(10):
        classLst.append({'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0})
    tp, tn, fp, fn = 0, 0, 0, 0

    # The predict_classes function outputs the highest probability class
    # according to the trained classifier for each input example.
    predicted_classes = model.predict_classes(x_test)

    for i in range(10000):
        row = y_test[i]
        index = int(row[predicted_classes[i]])
        if index == 1:
            tp += 1
            tn += 1
            classLst[predicted_classes[i]]['tp'] += 1
            classLst[predicted_classes[i]]['tn'] -= 1
        else:
            fp += 1
            fn += 1
            counter = 0
            for j in row:
                if int(j) == 1:
                    index = counter
                    break
                counter += 1
            classLst[predicted_classes[i]]['fp'] += 1
            classLst[index]['fn'] += 1
            classLst[predicted_classes[i]]['tn'] -= 1
            classLst[index]['tn'] -= 1
        for k in range(10):
            classLst[k]['tn'] += 1


    def prf1(class_dict):
        pp = class_dict['tp'] / (class_dict['tp'] + class_dict['fp'])
        rr = class_dict['tp'] / (class_dict['tp'] + class_dict['fn'])
        f11 = 2 * (pp * rr) / (pp + rr)
        return pp, rr, f11


    digit = 0
    for d in classLst:
        result = prf1(d)
        d['precision'] = result[0]
        d['recall'] = result[1]
        d['f1-score'] = result[2]
        print(digit, '==> Precision:', result[0], 'Recall:', result[1], 'F1-score:', result[2])
        digit += 1

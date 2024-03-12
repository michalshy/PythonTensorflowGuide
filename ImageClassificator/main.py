import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


def plot_images(i, predictionsArray, trueLabel, img):
    trueLabel, img = trueLabel[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    predictedLabel = np.argmax(predictionsArray)
    if(predictedLabel == trueLabel):
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predictedLabel],
                                         100*np.max(predictionsArray),
                                         class_names[trueLabel]),
               color = color)

def plotValueArr(i, predictionsArray, trueLabel):
    trueLabel = trueLabel[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisPlot = plt.bar(range(10), predictionsArray, color = "#777777")
    plt.ylim([0,1])
    predictedLabel = np.argmax(predictionsArray)

    thisPlot[predictedLabel].set_color("red")
    thisPlot[trueLabel].set_color("blue")


fashionMNIST = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashionMNIST.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneaker', 'Bag', 'Ankle boot']
#
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs = 10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probabilityModel = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probabilityModel.predict(test_images)

# numRows = 5
# numCols = 5
# numImages = numRows * numCols
#
# plt.figure(figsize=(2*2*numCols, 2*numRows))
# for i in range(numImages):
#     plt.subplot(numRows, 2*numCols, 2*i+1)
#     plot_images(i, predictions[i], test_labels, test_images)
#     plt.subplot(numRows, 2*numCols, 2*i+2)
#     plotValueArr(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()

img = test_images[1]
print(img.shape)

img = (np.expand_dims(img, 0))
print(img.shape)

predictionsSingle = probabilityModel.predict(img)
print(predictionsSingle)


plotValueArr(1, predictionsSingle[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()


np.argmax(predictionsSingle[0])
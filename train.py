# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from CNN import CNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os

def load_split(csvPath):
	# initialize the list of data and labels
	data = []
	labels = []
	# load the contents of the CSV file, remove the first line (since
	# it contains the CSV header), and shuffle the rows (otherwise
	# all examples of a particular class will be in sequential order)
	rows = open(csvPath).read().strip().split("\n")[1:]
	random.shuffle(rows)

	# loop over the rows of the CSV file
	for (i, row) in enumerate(rows):
		# check to see if we should show a status update
		if i > 0 and i % 1000 == 0:
			print("[INFO] processed {} total images".format(i))
		# split the row into components and then grab the class ID
		# and image path

		(label, imagePath) = row.strip().split(",")[-2:]
		# derive the full path to the image file and load it


		image = io.imread("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/"+imagePath) # basepath + image path directly


		data.append(image)
		labels.append(int(label))

	# convert the data and labels to NumPy arrays
	data = np.array(data, dtype=object)
	labels = np.array(labels, dtype =object)

	# return a tuple of the data and labels
	return (data, labels)


# initialize the number of epochs to train for, base learning rate,
# and batch size
NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 64

# load the label names
labelNames = open("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

print("[INFO] loading training and testing data...")
(trainX, trainY) = load_split("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/Train.csv")
(testX, testY) = load_split("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/Test.csv")



for i in range(len(trainX)):
	for j in range(len(trainX[i])):
		for k in range(len(trainX[i][j])):
			trainX[i][j][k] = trainX[i][j][k].astype("float32")/255.0

#trainX = trainX.astype("float32") / 255.0
print("training done")

for i in range(len(testX)):
	for j in range(len(testX[i])):
		for k in range(len(testX[i][j])):
			testX[i][j][k] = testX[i][j][k].astype("float32")/255.0
#testX = testX.astype("float32") / 255.0
print("testing done")

numLabels = len(np.unique(trainY))
trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = CNN.build(width=32, height=32, depth=3,
	classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")

print("[INFO] training network...")
#H = model.fit(aug.flow(trainX, trainY, batch_size=BS),

H = model.fit(trainX, trainY, batch_size=BS,
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BS,
	epochs=NUM_EPOCHS,
	verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))
# save the network to disk
print("[INFO] serializing network to '{}'...".format("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/output"))
model.save("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/output")


# plot the training loss and accuracy
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/output")

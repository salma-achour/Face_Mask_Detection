import tensorflow as tf
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


learning_rate = 0.0001
nb_epochs = 8
batch_size = 32

directory = r"Data"
categories = ["with_mask", "without_mask"]

print("[INFO] loading images...")

data = []
labels = []

for category in categories:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    	image = tf.keras.preprocessing.image.img_to_array(image)
    	image = preprocess_input(image)
    	data.append(image)
    	labels.append(category)

# Data cleaning
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# Data augmentation
aug = tf.keras.preprocessing.image.ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load pre-trained model
baseModel = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False,
	input_tensor= tf.keras.layers.Input(shape=(224, 224, 3)))

# create model
x = baseModel.output
x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
x = tf.keras.layers.Flatten(name="flatten")(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(2, activation="softmax")(x)
model = Model(inputs=baseModel.input, outputs=x)

# freeze pretrained model
for layer in baseModel.layers:
	layer.trainable = False

# compile  model
print("[INFO] compiling model...")
opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=learning_rate / nb_epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# tensorboard for monitoring metrics and gpu usage
tensorboard_callback = TensorBoard(log_dir="logs/{}".format(time()))

# train model
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=batch_size),
	steps_per_epoch=len(trainX) // batch_size,
	validation_data=(testX, testY),
	validation_steps=len(testX) // batch_size,
	epochs=nb_epochs,
    callbacks=[tensorboard_callback])

# evaluate model
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=batch_size)

# map prediction to label
predIdxs = np.argmax(predIdxs, axis=1)

# classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# save model for future usage
print("[INFO] saving mask detector model...")
model.save("model.model", save_format="h5")

# plot the training loss and accuracy
N = nb_epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("evaluation_plot.png")
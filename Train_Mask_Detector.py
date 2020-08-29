# First import all the Requirements

# Data Preprocessing
import tensorflow
import keras
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D , Dropout,Flatten , Dense , Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array , load_img
from keras.utils import to_categorical
import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR =  1e-4
EPOCHS = 20
BS = 32


DIRECTORY = "dataset"
CATEGORIES = ["with_mask" , "without_mask"]

print(" loading images....")


data , labels = [] , []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY , category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path , target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)
        
        data.append(image)
        labels.append(category)
        
# converting the image into the categorical variables

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# converting into the numpy arrays

data = np.array(data , dtype="float32")
labels = np.array(labels)

# Training the data using the train_test_split

(X_train , X_test , y_train , y_test) = train_test_split(data , labels , test_size = 0.20 , stratify = labels , random_state = 42)


# Data Augmentation

aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
# instead of convnet we here use mobilenet

baseModel = MobileNetV2(weights="imagenet" , include_top=False , 
                        input_tensor=Input(shape=(224,224,3)))



# creating the head model object

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(128 , activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2,activation="softmax")(headModel)


model = Model(inputs = baseModel.input , outputs = headModel)

for layer in baseModel.layers:
    layer.trainable = False
    


# compliing the model
    
print( "compiling model.....")

opt = Adam(Lr = INIT_LR ,decay= INIT_LR/EPOCHS)

model.compile(Loss= "binary_crossentropy" , optimizer=opt,
              metrics = ["accuracy"])


# train the head model

print("training head...")

H = model.fit(
        aug.flow(X_train , y_train ,batch_size=BS),
        steps_per_epoch=len(X_train) // BS,
        validation_data=(X_test , y_test),
        validation_steps=len(X_test) // BS,
        epochs=EPOCHS)


predIdxs = model.predict(X_test, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(y_test.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
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
plt.savefig("plot.png")











        

























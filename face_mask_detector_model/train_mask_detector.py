# USAGE
# python train_mask_detector.py --dataset ./dataset 

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Model Hyper Parameters
# initialize the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32




def load_dataset(dataset_folder_path):
    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading images...")
    return list(paths.list_images(dataset_folder_path))


def preprocess_dataset(imagePaths):
    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # load the input image (224x224) and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    return data, labels


def split_dataset_to_train_set_and_test_set(data, labels):
    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    return train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42), lb.classes_


def build_model():
    # load the MobileNetV2 network, ensuring the head FC layer sets are left off
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

    # construct the head of the model that will be placed on top of the the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    # place the head FC model on top of the base model
    # (this will become the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    return model


def get_input_augmentator():
    # construct the training image generator for data augmentation
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")


def train_model(model, trainX, trainY, testX, testY, batch_size=32, epochs=5, initial_lr=1e-4):
    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(lr=initial_lr, decay=initial_lr/epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # get the input augmentation feeder
    with tf.device("cpu:0"):
        aug = get_input_augmentator()

    # train the head of the network
    print("[INFO] training head...")
    return model.fit(
        aug.flow(trainX, trainY, batch_size=batch_size),
        steps_per_epoch=len(trainX) // batch_size,
        validation_data=(testX, testY),
        validation_steps=len(testX) // batch_size,
        epochs=epochs)


def evaluate_model(model, testX, testY, classes, batch_size=32):
    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=batch_size)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs, target_names=classes))


def save_model_to_disk(model, model_save_dir, model_save_name, save_format_h5=True, save_format_tf=True, save_weights=True):
    # serialize the model to disk (save_format='h5' or 'tf')
    print("[INFO] saving mask detector model...")
    
    # check if model_save_dir exists and if not create it
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    if save_format_h5:
        model.save(os.path.join(model_save_dir, model_save_name+'.h5'), save_format='h5')
        
    if save_format_tf:
        model.save(os.path.join(model_save_dir, model_save_name), save_format='tf')
    
    # Save the weights
    if save_weights:
        model.save_weights(os.path.join(model_save_dir, 'checkpoints', model_save_name))

    # save_path = os.path.join(model_save_dir, model_save_name, '1/')
    # tf.saved_model.save(model, save_path)

    from tensorflow.keras import backend as K
    tf.io.write_graph(K.get_session().graph, model_save_dir, model_save_name+'.pbtxt', as_text=True)


def save_model_loss_and_accuracy_graphs(H, graph_filename, epochs):
    # plot the training loss and accuracy
    if tf.__version__.split('.')[0] == '2':
        accuracy_key = 'accuracy'
        val_accuracy_key = 'val_accuracy'
    else:
        accuracy_key = 'acc'
        val_accuracy_key = 'val_acc'

    N = epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history[accuracy_key], label="train_acc")
    plt.plot(np.arange(0, N), H.history[val_accuracy_key], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(graph_filename)



def main(dataset, model_save_dir, model_save_name, model_eval_graph_name, train_on_cpu):
    # Force Use CPU
    if train_on_cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        tf.keras.backend.set_session(tf.Session(config=config));

    # grab the list of images in our dataset directory
    imagePaths = load_dataset(dataset)

    # pre-process dataset and extract the class label from the filename
    data, labels = preprocess_dataset(imagePaths)

    # partition the data into training and testing splits and perform one-hot encoding on the labels
    (trainX, testX, trainY, testY), classes = split_dataset_to_train_set_and_test_set(data, labels)

    # choose device to use for execution
    device_name = "cpu:0" if train_on_cpu else "gpu:0"
    with tf.device(device_name):
        # build model
        model = build_model()

        # train model
        H = train_model(model, trainX, trainY, testX, testY, batch_size=BS, epochs=EPOCHS, initial_lr=INIT_LR)

        # evaluate model
        evaluate_model(model, testX, testY, classes, batch_size=BS)

    # save model
    save_model_to_disk(model, model_save_dir, model_save_name)

    # save graph
    save_model_loss_and_accuracy_graphs(H, model_eval_graph_name, epochs=EPOCHS)
    
    return model, H



if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=False, 
		default="./dataset",
        help="path to input dataset")
    ap.add_argument("-p", "--plot", type=str, 
		default="face_mask_detector_train_graph.png",
        help="path to output loss/accuracy plot")
    ap.add_argument("-s", "--model-dir", type=str,
        default="./face_mask_detector",
        help="path to output face mask detector model root directory")
    ap.add_argument("-m", "--model", type=str,
        default="face_mask_detector_model",
        help="name to output face mask detector model")
    ap.add_argument("-c", "--train-on-cpu", dest='cpu', action='store_true',
        default=False,
        help="use to force use CPU for model training")
    args = vars(ap.parse_args())

    main(args["dataset"], args["model-dir"] args["model"], args["plot"], args['cpu'])


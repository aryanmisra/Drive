import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import itertools
from keras import metrics
import matplotlib.pyplot as plt
from keras.models import model_from_json
import os
from keras.models import load_model

K.tensorflow_backend._get_available_gpus()

train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'

DIR1 = "base_dir/train_dir/forward"
DIR2 = "base_dir/train_dir/left"
DIR3 = "base_dir/train_dir/right"

# Declare a few useful values
num_train_samples = (len([name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))])) + (len([name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))])) + (len([name for name in os.listdir(DIR3) if os.path.isfile(os.path.join(DIR3, name))]))
num_val_samples = ((len([name for name in os.listdir(DIR1.replace("train_dir","val_dir")) if os.path.isfile(os.path.join(DIR1.replace("train_dir","val_dir"), name))])) + (len([name for name in os.listdir(DIR2.replace("train_dir","val_dir")) if os.path.isfile(os.path.join(DIR2.replace("train_dir","val_dir"), name))])) + (len([name for name in os.listdir(DIR3.replace("train_dir","val_dir")) if os.path.isfile(os.path.join(DIR3.replace("train_dir","val_dir"), name))])))
train_batch_size = 10
print(str(num_train_samples) + " NUM TRAINING SAMPLES")
print(str(num_val_samples) + " NUM VAL SAMPLES")

val_batch_size = 10
image_x = 224
image_y = 224
image_channels = 3

# Declare how many steps are needed in an iteration
train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

# Set up generators
train_batches = ImageDataGenerator(
    rotation_range=60,
    zoom_range=[0,2],
    
    preprocessing_function= \
        keras.applications.mobilenet.preprocess_input).flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=train_batch_size)

valid_batches = ImageDataGenerator(
    rotation_range=60,
    zoom_range=[0,2],
    
    preprocessing_function= \
        keras.applications.mobilenet.preprocess_input).flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=val_batch_size)

test_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.mobilenet.preprocess_input).flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=val_batch_size,
    shuffle=False)
print(test_batches)
# Create a MobileNet model
mobile = keras.applications.mobilenet.MobileNet(weights='imagenet')
x = mobile.layers[-6].output
x = Dropout(0.15)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=predictions)

for layer in model.layers[:-26]:
    layer.trainable = False

model.summary()

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy', metrics.categorical_accuracy])

# Add weights to make the model more sensitive to melanoma
class_weights={
    0: 1.0,  # forward
    1: 2.0,  # left
    2: 2.0,  # rights

}

filepath = "saves/model.h5"
# Declare a checkpoint to save the best version of the model
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

# Reduce the learning rate as the learning stagnates
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                              verbose=1, mode='min', min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]
if os.path.exists(filepath):
    model = load_model(filepath)
# Fit the model
history = model.fit_generator(train_batches,
                              steps_per_epoch=train_steps,
                              class_weight=class_weights,
                              validation_data=valid_batches,
                              validation_steps=val_steps,
                              epochs=1000,
                              verbose=1,
                              callbacks=callbacks_list)

"""
val_loss, val_cat_acc = model.evaluate_generator(test_batches, steps=val_steps)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)

# Evaluation of the best epoch
model.load_weights(filepath)
val_loss, val_cat_acc = \
model.evaluate_generator(test_batches, steps=val_steps)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)


# Create a confusion matrix of the test images
test_labels = test_batches.classes

# Make predictions
predictions = model.predict_generator(test_batches, steps=val_steps, verbose=1)

# Declare a function for plotting the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

cm_plot_labels = ['forward', 'left', 'right']

plot_confusion_matrix(cm, cm_plot_labels)
"""
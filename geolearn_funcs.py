# @Author: Martin Blouin <mablou>
# @Date:   2019-02-25T12:30:09-05:00
# @Email:  martin.blouin@geolearn.ca
# @Last modified by:   mablou
# @Last modified time: 2019-03-01T11:10:59-05:00



import numpy as np
import os
from keras.utils import Sequence,to_categorical
import keras
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt


def set_up_project(data_path):
    try:
        if ('LEM-37' not in os.listdir(data_path)) \
                    or ('LEM-18' not in os.listdir(data_path)):
            print('Path error - Please specify good data path ')
        else:
            print('Project setup OK ')
    except:
        print('Path error - Please specify good data path ')


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3,3,figsize=(4,9))
    fig.subplots_adjust(hspace=0.3, wspace=2)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i])

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0},\n Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
##########
#########

class DataPipeline(Sequence):
    def __init__(self, data_dir,batch_size=8, train=False):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.train = train
        self.LABELS_DICT = np.array(['gabbro',
                                      'diorite',
                                      'QFP',
                                      'rhyolite',
                                      'andésite'])
        self.load_data()

    def __len__(self):
        return int(len(self.df) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for i in np.arange(idx * self.batch_size, (idx + 1) * self.batch_size):

            x, y = self.generate_xy(i)
            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x),np.array(batch_y)

    def generate_xy(self, idx):

        if self.train:
            file_path = os.path.join('Export_10cm_slices',
                                     'LEM-37_' + str(self.df['depth_top'][idx])+'.0.jpeg')
            photo_path = os.path.join(self.data_dir,'LEM-37')

        else:
            file_path = os.path.join('Export_10cm_slices',
                                     'LEM-18_' + str(self.df['depth_top'][idx])+'.0.jpeg')
            photo_path = os.path.join(self.data_dir,'LEM-18')

        photo = imread(os.path.join(photo_path,file_path))
        photo = photo / 255
        label = to_categorical(self.df['labels'][idx], num_classes=5)
        return resize(photo,(200,40),anti_aliasing=True),label
        # return np.pad(photo[200:424],((0,0),(12,12),(0,0)),mode='constant'), label

    def load_data(self):
        if self.train:
            self.df = pd.read_csv(os.path.join(self.data_dir,
                                                   'LEM-37/LEM-37_labels.csv'))
        else:
            self.df = pd.read_csv(os.path.join(self.data_dir,
                                                   'LEM-18/LEM-18_labels.csv'))

        self.df['labels'] = self.encode_labels(self.df['lithology'])


    def encode_labels(self,labels):
        return np.argmax(labels.values.reshape(1,-1)==(self.LABELS_DICT).reshape(-1,1),axis=0)

def plot_examples(data_dir,train,prediction=False,model=None):

    data_pipeline = DataPipeline(data_dir,1,train)
    photos = []
    labels = []
    for _ in range(9):
        idx = np.random.randint(len(data_pipeline.df))
        p,l= data_pipeline[idx]
        photos.append(p[0])
        labels.append(data_pipeline.df['lithology'][idx])
    if prediction:
        pred = np.argmax(make_prediction(model,np.array(photos)),axis=1)
        pred_labels = data_pipeline.LABELS_DICT[pred]
        plot_images(np.array(photos),np.array(labels),pred_labels)

    else:
        plot_images(np.array(photos),np.array(labels))

def plot_histogram(data_dir,train):
    dp = DataPipeline(data_dir,train=train)
    plt.hist(dp.df.labels)
    plt.xticks(range(5),dp.LABELS_DICT)

def make_prediction(model,photos):
    return model.predict(photos)

def build_model(type,input_shape,loss='categorical_crossentropy'):
    if type.lower() == 'simple':
        model = simple_net(input_shape)

    adam = keras.optimizers.Adam()
    model.compile(optimizer=adam,
                  loss=loss,
                  metrics=['accuracy'])

    return model

def simple_net(input_shape):
    input = keras.layers.Input(shape=input_shape)
    encoder = keras.layers.Conv2D(64,(3,3),padding='same')(input)
    encoder = keras.layers.Activation('relu')(encoder)
    encoder = keras.layers.GlobalMaxPooling2D()(encoder)
    encoder = keras.layers.Dense(64)(encoder)
    encoder = keras.layers.Activation('relu')(encoder)
    encoder = keras.layers.Dense(5)(encoder)
    encoder = keras.layers.Activation('softmax')(encoder)

    return keras.Model(inputs=input,outputs=encoder)

def train_model(data_dir,model,epochs):
    tb = keras.callbacks.TensorBoard(log_dir='./tensorboard-logs',
                                   histogram_freq=0,
                                   write_graph=True,
                                   write_images=True)

    model.fit_generator(generator=DataPipeline(data_dir,train=True),
                        epochs=epochs,
                        validation_data=DataPipeline(data_dir,train=False),
                        shuffle=True,
                        verbose=1,
                        callbacks=[tb])

def get_report(data_dir,model):
    y_pred = model.predict_generator(DataPipeline(data_dir,train=False))
    y_pred = np.argmax(y_pred,axis=1)
    y_true = DataPipeline(data_dir,train=False).df['labels']
    print(classification_report(y_true[:-2],y_pred))

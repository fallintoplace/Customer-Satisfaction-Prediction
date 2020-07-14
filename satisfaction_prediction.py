import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib import dates as md
from tensorflow import keras
from tensorflow.keras import layers, losses, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1DTranspose, Conv1D, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

KEY = 'ID_code'
TARGET = 'target'
TRAIN_URL = "C:\\Users\\Minh\\Downloads\\santander-customer-transaction-prediction\\train.csv"
FINAL_URL = "C:\\Users\\Minh\\Downloads\\santander-customer-transaction-prediction\\test.csv"
RESULT_URL = "C:\\Users\\Minh\\Downloads\\santander-customer-transaction-prediction\\result.csv"
NROWS = 50000

"""
PREPARING THE DATASET

"""


dataset = pd.read_csv(TRAIN_URL)
dataset = dataset.dropna()
dataset.pop(KEY)

dataset = dataset.sample(frac=1).reset_index(drop=True)

train_dataset = dataset.sample(frac=1,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop(TARGET)
test_labels = test_dataset.pop(TARGET)

final = pd.read_csv(FINAL_URL)
final_key = final.pop(KEY)


train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
final = norm(final)
test_dataset = norm(test_dataset)
train_dataset = norm(train_dataset)

print(train_dataset.head())

"""
TWO DENSELY PRELU ACTIVATED CONNECTED LAYERS WITH DROPOUTS, REGULARIZATIONS
AND LAYER NORMALIZATIONS

"""

def build_model():
    model = Sequential([
        layers.Dense(
            50, input_shape=(test_dataset.shape[-1],),
        ),
        layers.PReLU(alpha_initializer=tf.initializers.constant(0.25)),
        layers.LayerNormalization(),
        layers.Dropout(0.5),
        layers.Dense(50),
        layers.PReLU(alpha_initializer=tf.initializers.constant(0.25)),
        layers.LayerNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
        ])

    metrics = [
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name='auc'),
    ]
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.0005), loss="binary_crossentropy", metrics=metrics
    )
    


    return model

model = build_model()


"""
FITTING AND PLOTTING THE LOSS VALUE GRAPH

"""

counts = np.bincount(train_labels)
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(train_labels)
    )
)

weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]



class_weight = {0: weight_for_0, 1: weight_for_1}
history = model.fit(
    train_dataset,
    train_labels,
    batch_size=256,
    epochs=300,
    verbose=1,
    callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_auc', patience = 10)],
    validation_split = 0.2,
    class_weight=class_weight,
)
plt.ylim([0, 1])
plt.plot(history.history["auc"], label="Training AUC")
plt.plot(history.history["val_auc"], label="Validation AUC")
plt.legend()
plt.show()


"""
PRINTING THE RESULT

"""


# result = pd.DataFrame()
# result[KEY] = test_labels
# result[TARGET] = model.predict(test_dataset).flatten()
# #result[TARGET] = result[TARGET].apply(lambda x: 0 if (x<0.5) else 1)
# pd.DataFrame(result).to_csv(RESULT_URL, index = False)

result = pd.DataFrame()
result[KEY] = final_key
result[TARGET]=model.predict(final).flatten()
result[TARGET]=result[TARGET].apply(lambda x: 0 if (x<0.5) else 1)
pd.DataFrame(result).to_csv(RESULT_URL, index = False)

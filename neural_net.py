
import glob
import numpy as np
import pickle
from make_ligand_block import LigandBlock
from make_interaction_gbsa_block import GbsaInteraction
from make_complex_gbsa_block import GbsaComplexBlock
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
seed_value= 42
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import normalization
from keras.layers.normalization import BatchNormalization

# read pickle file
def load_pickle(name):
    x = pickle.load(open(name, 'rb'))
    return x



features = load_pickle("neural_net_features_09_20_23.pickle")
regression_label = load_pickle("exp_binding_energy.pickle")
species = [x for x in regression_label if x in features]
ar_features = np.array([features[x] for x in species])
ar_regression_label = np.array([regression_label[x] for x in species])
x = np.reshape(ar_features, (len(species), 39))
y = np.reshape(ar_regression_label, (len(species), 1))
x_df = pd.DataFrame(x)
y_df = pd.DataFrame(y)
standard_scaler = preprocessing.StandardScaler()
x_scaled = standard_scaler.fit_transform(x_df)
x_scaled = pd.DataFrame(x_scaled)
x = x_scaled.iloc[:,:].values
y = y_df.iloc[:,:].values
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.1, random_state=42)


model = keras.Sequential()
model.add(BatchNormalization())
model.add(Dense(64, input_dim=39, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation=None))
model.compile(loss='mean_squared_error', optimizer='adamax')
history = model.fit(X_train, y_train, epochs=200, batch_size=128)
y_pred = model.predict(X_test)

linear_model = LinearRegression()
linear_model.fit(y_pred, y_test)
r_squared = linear_model.score(y_pred, y_test)
print(r_squared)
plt.scatter(y_pred, y_test)
plt.show()

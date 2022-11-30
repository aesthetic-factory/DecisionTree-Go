import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

import pandas as pd

def create_model():
    main_input_1 = Input(shape=(3),name='main_input_1')
    hidden_layer = BatchNormalization()(main_input_1)
    
    hidden_layer = Dense(16, activation="relu", name="dense_1")(hidden_layer)
    hidden_layer = BatchNormalization()(hidden_layer)

    hidden_layer = Dense(8, activation="relu", name="dense_2")(hidden_layer)
    hidden_layer = BatchNormalization()(hidden_layer)

    hidden_layer = Dense(1, activation="relu", name="dense_3")(hidden_layer)

    output = Activation('sigmoid', name='output')(hidden_layer)

    model = Model(inputs=main_input_1, outputs=output)

    # opt = Adam(learning_rate=4e-03, epsilon=4e-04)
    # model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# main
model = create_model()

df = pd.read_csv('testing_data\data.csv')  
y_train = df[['label']].to_numpy()
# df['salary'] = df['salary'] / 30000
# df['browow'] = df['browow'] / 30000

x_train = df[['salary','browow','dmp1']].to_numpy()

model.summary()

def scheduler(epoch, lr):
    if epoch < 40:
        return lr
    else:
        return lr * tf.math.exp(-0.3)

#tensorboard = TensorBoard(log_dir=".\logs\{}".format(model_name))
learningRateScheduler = LearningRateScheduler(scheduler)
# checkpoint = ModelCheckpoint(f".\models\{model_name}.h5", monitor='val_loss',
#                                 verbose=0, save_best_only=True, mode='min')  # saves only the best ones
history = model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.2, callbacks=[learningRateScheduler])
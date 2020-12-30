import tensorflow as tf
import pandas as pd
import matplotlib as plt
from tqdm import tqdm
import pickle as pkl
import numpy as np

mat_val_dict=None
with open("train_data/mat_val_dict.pkl","rb") as p:
    mat_val_dict=pkl.load(p)

def NCWH_2_NHWC(mat):
    return tf.transpose(mat,[0,2,3,1])

def map_id_2_val(id):
    global mat_val_dict
    return [mat_val_dict[id]/255]

def str_2_lst(str_):
    return list(map(float,str_.split()))

train_df=pd.read_csv("train_data/train_dat.csv")

#load data to np
train_x_id=train_df["id"].values
train_x_h=train_df["height"].values
train_x_w=train_df["width"].values

train_y=train_df["label"].values

#format data
train_x_img_val=np.asarray(list(map(map_id_2_val,train_x_id)),dtype=np.float32)

train_y=np.asarray(list(map(str_2_lst,train_y)),dtype=np.float32)

#Model
val_input_layer=tf.keras.Input(name="val_input",shape=(1,256,256))
h_input_layer=tf.keras.Input(name="height_input",shape=(1,1))
w_input_layer=tf.keras.Input(name="width_input",shape=(1,1))


#pixel layer
reshape_layer=tf.keras.layers.Reshape((256,256,1))(val_input_layer)
conv_1_layer=tf.keras.layers.Conv2D(3,(3,3),name="conv1",activation='relu',input_shape=(256,256,1))(reshape_layer)
mpool_1_layer=tf.keras.layers.MaxPool2D((5,5),name="mpool1")(conv_1_layer)
conv_2_layer=tf.keras.layers.Conv2D(3,(3,3),name="conv2",activation='relu')(mpool_1_layer)
mpool_2_layer=tf.keras.layers.MaxPool2D((5,5),name="mpool2")(conv_2_layer)
flatten_layer=tf.keras.layers.Flatten(name="flat_img")(mpool_2_layer)

#h-w layer
re_flat_layer=tf.keras.layers.Reshape((1,243),name="reshape_flat")(flatten_layer)

concat_h=tf.keras.layers.Concatenate(name="concat_h")([re_flat_layer,h_input_layer])
concat_w=tf.keras.layers.Concatenate(name="concat_w")([concat_h,w_input_layer])
sum_flat_layer=tf.keras.layers.Flatten(name="summary_flat")(concat_w)

#final-Dense
den_1_layer=tf.keras.layers.Dense(256,activation='relu',name="den1")(sum_flat_layer)
final_output=tf.keras.layers.Dense(4,activation='relu',name="output")(den_1_layer)

model = tf.keras.Model(inputs=[val_input_layer,h_input_layer,w_input_layer], outputs=final_output,name="random_fame_model")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=tf.keras.losses.MeanSquaredError())
model.fit(x=[train_x_img_val,train_x_h,train_x_w],y=train_y,verbose=1,epochs=130,batch_size=32)
model.save("rnd_fame_model.h5")
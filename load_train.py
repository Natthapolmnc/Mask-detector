import tensorflow as tf
import pandas as pd
import cv2 as cv
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

model = tf.keras.models.load_model("rnd_fame_model.h5")

model.fit(x=[train_x_img_val,train_x_h,train_x_w],y=train_y,verbose=1,epochs=100,batch_size=32)
model.save("rnd_fame_model.h5")
import tensorflow as tf
import pandas as pd
import cv2 as cv
import matplotlib as plt
from tqdm import tqdm
import pickle as pkl
import numpy as np

mat_val_dict=None
with open("test/test_mat_val_dict.pkl","rb") as p:
    mat_val_dict=pkl.load(p)

def map_id_2_val(id):
    global mat_val_dict
    return [mat_val_dict[id]/255]

def str_2_lst(str_):
    return list(map(float,str_.split()))

test_df=pd.read_csv("test/test_dat.csv")
test_x_id=test_df["id"].values
test_x_h=test_df["height"].values
test_x_w=test_df["width"].values

test_x_img_val=np.asarray(list(map(map_id_2_val,test_x_id)),dtype=np.float32)

model = tf.keras.models.load_model("rnd_fame_model_plus_.h5")


y_sum=model.predict([test_x_img_val,test_x_h,test_x_w])
print (y_sum)
for i in range(len(y_sum)):
    image=cv.imread("test/Face-Mask"+str(test_x_id[i])+".jpg")
    drw_img=cv.rectangle(image,(y_sum[i][0],y_sum[i][1]),(y_sum[i][0]+y_sum[i][2],y_sum[i][1]+y_sum[i][3]),(0,100,255),2)
    cv.imwrite("test/output/out_"+str(test_x_id[i])+".jpg",drw_img)


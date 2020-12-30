import pandas as pd
import json
import pickle as pkl
import numpy as np
import cv2 as cv

data=open("data_preprocess/label_data.csv","r")

for i in data:
    lst_dat=i.split(",")
    image=cv.imread("data_preprocess/train/"+lst_dat[5])
    g=list(map(int,lst_dat[1:5]))
    drw_img=cv.rectangle(image,(int(g[0]),int(g[1])),(int(g[0]+g[2]),int(g[1]+g[3])),(255,255,0))
    cv.imwrite("data_preprocess/train/test_label/labe_"+lst_dat[5],drw_img)
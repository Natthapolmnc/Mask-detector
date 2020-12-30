import pandas as pd
import numpy as np
import pickle as pkl
import cv2 as cv
import json

img_val_dic={}


with open("test_label_data.csv","r") as file:
    res_dict={}
    res_dict["id"]=[]
    res_dict["height"]=[]
    res_dict["width"]=[]
    res_dict["label"]=[]
    
    indx_=0
    for i in file:
        lst_dat=i.split(",")
        bbox=list(map(int,lst_dat[1:5]))
        w=int(lst_dat[6])
        h=int(lst_dat[7])
        res_dict["label"].append(" ".join(list(map(str,bbox))))
        res_dict["width"].append(w)
        res_dict["height"].append(h)
        res_dict["id"].append(indx_)
        gry_mat=cv.imread("test/"+lst_dat[5],0)
        gry_mat=cv.resize(gry_mat,(256,256))
        img_val_dic[indx_]=gry_mat
        indx_+=1


mat_file=open("test_mat_val_dict.pkl","wb")
pkl.dump(img_val_dic,mat_file)
mat_file.close()


train_x=pd.DataFrame(data=res_dict)
train_x.to_csv("train_dat.csv",index=False)
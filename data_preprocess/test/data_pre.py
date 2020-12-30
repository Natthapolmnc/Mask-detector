import pandas as pd
import numpy as np
import pickle as pkl
import cv2 as cv
import json

img_val_dic={}
res_dict={}
res_dict["id"]=[]
res_dict["height"]=[]
res_dict["width"]=[]
res_dict["label"]=[]

data=open("test_dat.json","r")
json_dat=json.load(data)
img_json=json_dat["images"]
label_json=json_dat["annotations"]

for i in range(len(img_json)):
    gry_mat=cv.imread(img_json[i]["file_name"],0)
    gry_mat=cv.resize(gry_mat,(256,256))
    img_val_dic[img_json[i]["id"]]=np.array(gry_mat,np.float32)
    res_dict["id"].append(img_json[i]["id"])
    res_dict["height"].append(img_json[i]["height"])
    res_dict["width"].append(img_json[i]["width"])
    res_dict["label"].append(" ".join(list(map(str,label_json[i]["bbox"]))))




mat_file=open("test_mat_dict.pkl","wb")
pkl.dump(img_val_dic,mat_file)
mat_file.close()
data.close()


train_x=pd.DataFrame(data=res_dict)
train_x.to_csv("test_dat.csv",index=False)





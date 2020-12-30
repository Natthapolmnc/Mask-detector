import os


file_n_fol=os.listdir("train")

def int_2(_int):
    if _int//100>=1:
        return str(_int)
    if _int//10>=1:
        return "0"+str(_int)
    return "00"+str(_int)

for i in range(len(file_n_fol)):
    os.rename("train/"+file_n_fol[i],"train/train_image_"+int_2(i)+".jpg")

import easyocr
from pdf2image import convert_from_path
from tqdm import tqdm
import os
import shutil
import re
import sys
import glob
import os
import io
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='data/json/proven-cider-382907-953301e26635.json'
from google.cloud import vision
import re

vision_client = vision.ImageAnnotatorClient()

folder_list=[]
for root, folders, files in os.walk('data/img/'):
    for folder in folders:
        folder_list.append(os.path.join('data/img',folder))

img_root='data/img'
text_root='data/txt_by_gvision'
error_list='data/error_gvision.txt'
counter=1
total_folder=len(folder_list)
for root,folders,files in os.walk(img_root):
    if files:
        filenum=len(files)
        print('\r{} {}/{}'.format(root,counter,total_folder),end='')
        fn=os.path.split(root)[-1]  # get folder name to befurther assigned as txt name
        fn=fn+'.txt'
        txt_path=os.path.join(text_root,fn)
        if os.path.exists(txt_path):
            os.remove(txt_path)
        with open(txt_path,'w') as fo:
            for i,img_f in enumerate(files):
                img_fn=str(i)+'.jpg'
                img_fp=os.path.join(root,img_fn)
                try:
                    with io.open(img_fp, 'rb') as image_file:
                        content = image_file.read()
                    image = vision.Image(content=content)
                    response = vision_client.text_detection(image=image)
                    text = response.text_annotations[0].description
                    text=text.replace('\n',' ')
                    fo.write(text)
                    fo.write('\n')
                except Exception as x:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    with open(error_list,'a') as ferr:
                        ferr.write('{}======>{},{}'.format(txt_path,str(x),exc_tb.tb_lineno))
                        ferr.write('\n')
                # break
        # break
        counter+=1
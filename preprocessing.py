# PACKAGES
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pdb
import time 
import copy
import cv2
import gc



## inspired by 
# https://github.com/RobinSmits/KaggleBengaliAIHandwrittenGraphemeClassification/blob/master/KaggleKernelEfficientNetB3/preprocessing.py
# https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image


def normalize_and_resize_img(img, new_width,new_height):
    img = 255 -img

    img = (img * (255.0 / img.max())).astype(np.uint8)

    img = img.reshape(137,236)
    resized_image = cv2.resize(img, dsize=(new_width,new_height), interpolation=cv2.INTER_AREA)

    return resized_image

def resize_and_save_image(img,image_id,output_dir,original_width,original_height, new_width,new_height):

    ## invert and normalize first
    normalized_resized_image = normalize_and_resize_img(img,new_width,new_height)
    cv2.imwrite(output_dir + str(image_id) + '.png',normalized_resized_image)


def create_new_images(data_dir, output_dir, original_width,original_height,new_width, new_height):

    # parquet_locations = [ x for x in os.listdir(data_dir) if 'train_image_data' in x ]
    # parquet_locations.sort()


    for i in tqdm(range(0,4)):


        ## read parquet files
        parquet_data = pd.read_parquet(os.path.join(data_dir, 'train_image_data_'+str(i)+'.parquet'))

        image_ids = parquet_data['image_id'].values

        parquet_data = parquet_data.drop('image_id', axis='columns')

        for image_id, index in zip(image_ids, range(parquet_data.shape[0])):
            # image = parquet_data.iloc[index].values
            resize_and_save_image(parquet_data.iloc[index].values, image_id,output_dir, original_width, original_height, new_width,new_height)
        

        del parquet_data
        gc.collect()

create_new_images('../../data/bengaliai-cv19','./results/',236,137,256,256)

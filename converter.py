from collections import namedtuple
from object_detection.utils import dataset_util, label_map_util
from PIL import Image
import tensorflow.compat.v1 as tf
import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

# Dict of your classes 

dict = {"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"i":9,"j":10,"k":11,"l":12,"m":13,"n":14,"o":15,"p":16,"q":17,"r":18,"s":19,"t":20,"u":21,"v":22,"w":23,"x":24,"y":25,"z":26} 

def create_tf_example(item_number):

    with tf.gfile.GFile(str(item_number)+".jpg", 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = str(item_number)+".jpg"
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    file = open(str(itemnumber)+".txt","r")
    
    
    
    for row in file.readlines():
        
        #every line is one Boundingbox
        
        # print(row)
        xmins.append(float(row.split(" ")[1]))
        xmaxs.append(float(row.split(" ")[3]))
        ymins.append(float(row.split(" ")[2]))
        ymaxs.append(float(row.split(" ")[4]))
        classes_text.append(row.split(" ")[0].encode('utf-8'))
        classes.append(dict[row.split(" ")[0]])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

for item in os.listdir():
    
    if item.endswith(".jpg"):

        test = create_tf_example(item)

        writer = tf.python_io.TFRecordWriter("yahoo.record")
        writer.write(test.SerializeToString())

writer.close()

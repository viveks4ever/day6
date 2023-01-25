
import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
st.set_option('deprecation.showfileUploaderEncoding', False)
# Loading saved model from Drive.
from keras.models import load_model
model = load_model('MNISTANN.h5')

html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:50px;color:white;margin-top:10px;">Poornima Institute of Enginerring & Technology  </p></center> 
<center><p style="font-size:40px;color:white;margin-top:10px;"> Department of Computer Engineering</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Six Days Faculty development Program on Applications of Neural Network and Machine learning </p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Digit Recognition
         """
         )
file= st.file_uploader("Please upload image", type=("jpg", "png"))

import cv2
from  PIL import Image, ImageOps
def import_and_predict(image_data):
   
  image_resized = cv2.resize(image_data, (28, 28))  
  prediction = model.predict(image_resized.reshape(1,784))
  print('Prediction Score:\n',prediction[0])
  thresholded = (prediction>0.5)*1
  print('\nThresholded Score:\n',thresholded[0])
  print('\nPredicted Digit:',np.where(thresholded == 1)[1][0])
  digit = np.where(thresholded == 1)[1][0]
  #st.image(image_data, use_column_width=True)
  return digit
if file is None:
  st.text("Please upload an Image file")
else:
  image=Image.open(file)
  image=np.array(image)
  #file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  #image = cv2.imdecode(file_bytes, 1)
  st.image(image,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Predict Digit"):
  result=import_and_predict(image)
  st.success('Model has predicted the image is of  {}'.format(result))
if st.button("About"):
  st.header(" Deepak Moud")
  st.subheader("Head, Department of Computer Engineering")
  
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Developed by</p></center> 
   <center><p style="font-size:20px;color:white;margin-top:10px;">Deepak Moud</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)

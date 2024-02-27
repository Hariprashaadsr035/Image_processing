import streamlit as st
import os
import shutil
import util
import pandas as pd
from PIL import Image

tempDir = '/Users/alpha/Desktop/temp'

st.title('Image Classification using Machine Learning')
path_to_cr_data = tempDir

def save_uploadedfile(uploadedfile):
        with open(os.path.join(tempDir,uploadedfile.name),"wb") as f:
            f.write(uploadedfile.getbuffer())
            ret_str = f"Saved File:{uploadedfile.name} to tempDir"
        return st.success(ret_str)

@st.cache_resource
def load():
    util.load_saved_artifacts()

file = st.file_uploader('Upload a image')
if file is not None:
    l,r = st.columns(2)
    with l:
        image = Image.open(file)
        st.image(image)
        util.load_saved_artifacts()
        if os.path.exists(path_to_cr_data):
            shutil.rmtree(path_to_cr_data)
        os.mkdir(path_to_cr_data)
        save_uploadedfile(file)
        file_path = tempDir + '/' + file.name

    with r:
        res = util.classify_img(None,file_path = file_path)
        
        for j,i in enumerate(res):
            
            dic = {'Name' : i['class_dictionary'].keys(),
            'Probability': i['class_probability']}
            df = pd.DataFrame(dic)
            df = df.sort_values('Probability',ascending = False,ignore_index=True)
            if df.iloc[0,1] <40:
                 st.write('No Person Detected')
                 continue
            cls = res[j]['class']
            strg = f'The Face {j+1} is {cls.upper()}'
            st.write(strg)
            st.dataframe(df)
            

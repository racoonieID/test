import streamlit as st
import pandas as pd
import time
from PIL import Image

icon_image = Image.open('favicon.png')

st.set_page_config(
    page_title="Segmentasi Citra",
    page_icon=icon_image,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


st.title('Segmentasi Citra Optic Disk')

from io import StringIO

uploaded_file = st.file_uploader("Choose image file")
if uploaded_file is not None:
    # To read file as bytes:
    filename = uploaded_file.name.lower()
    if uploaded_file.name.endswith('jpg') or uploaded_file.name.endswith('png'):
        st.write('Accepted')


    # Can be used wherever a "file-like" object is accepted:
    # dataframe = pd.read_csv(uploaded_file)
    # st.write(dataframe)

import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps

st.title("使用谷歌的可教机器进行图像分类")
st.header("蚂蚁vs蜜蜂")
st.text("上传彩色蚂蚁,蜜蜂图片")

uploaded_file = st.file_uploader("选择..", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='上传了图片。', use_column_width=True)
    st.write("")
    st.write("分类...")
    label = teachable_machine_classification(image, 'keras_model.h5')
    if label == 0:
        st.write("蚂蚁")
    else:
        st.write("蜜蜂")
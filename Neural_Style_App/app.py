import streamlit as st
from PIL import Image
from stylize import style

st.title("StylizeMe")
st.write("Transform your image into any style of your choice using AI(Neural Style Transfer using VGG19)! The name 'StylizeMe' was suggested by OpenAI's ChatGPT.")

file = st.file_uploader("Upload your Content Image here .....",type=['jpg'])
styler = st.file_uploader("Upload your Style Image here .....",type=['jpg'])

if file is not None:

    image = Image.open(file)
    st.write("Original Image")
    st.image(image)
    st.write(file)

    style_image = Image.open(styler)
    st.write("Style Image")
    st.image(style_image)
    st.write(styler)

    button = st.button("Stylize")

    if button:
        st.write("It will take few minutes.....")
        generater = style(image,style_image)
        st.write("Output Image (300x400) - ")
        st.image(generater)

    
import streamlit as st
from PIL import Image
from stylize import style

st.title("Picassify")
st.write("Transform your image into the style of Picasso's paintings using AI(Neural Style Transfer using VGG19)! The name 'Picassify' was suggested by OpenAI's ChatGPT.")

file = st.file_uploader("Upload your Image here .....",type=['jpg'])
styler = Image.open('style.jpg')
st.write("Style Image")
st.image(styler)

if file is not None:

    image = Image.open(file)
    st.write("Original Image")
    st.image(image)
    st.write(file)

    button = st.button("Stylize")

    if button:
        st.write("It will take few minutes.....")
        generater = style(image,styler)
        st.write("Output Image (300x400) - ")
        st.image(generater)

    
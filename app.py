import streamlit as st
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title = "multiple pages",
    # page_icon = "🤣",
)
st.title("Main page")
st.sidebar.success("Select above page")
# profile picture 
def resize_image(image, size):
    resized_image = image.resize(size)
    return resized_image
image = Image.open("profilephoto.png")
size = (80,80)
a = resize_image(image,size)
st.image(a)

# title
# st.title("Ravi Chandera")

# # intro
# st.write("Hello there, I am pursuing masters in Artificial Intelligence at IIT patna. I have a strong foundation in advanced statistical analysis, machine learning, and deep learning techniques, enabling me to develop accurate and robust predictive models.")
# st.write("My working includes data preprocessing, feature engineering, and dimensionality reduction techniques to extract meaningful insights and improve model performance.")
# st.write("My programming background in Python helps to utilize libraries such as NumPy, Pandas, and scikit-learn to implement data science algorithms and perform data manipulation and visualization.")
# st.write("specialization in natural language processing (NLP) helps me to understand both traditional and state-of-the-art NLP techniques, allowing me to design and develop innovative solutions for language-related challenges")
# st.write("I know design and implementation of convolutional neural networks (CNNs) and advanced architectures such as ResNet, VGG, and Inception, allowing me to achieve state-of-the-art performance on image classification, object detection, and semantic segmentation tasks.")

# # course information
# df = pd.DataFrame({
#     "Courses taken" : ["Machine learning","Artificial intelligence","Deep learning","Natural language processing","Big Data","statistics and probability","Optimisation","Cyber security with Blockchain","Industry 4.0"]
#     })

# st.dataframe(df)
# write home page stuff here
st.sidebar.title("📜 educational information")
    # title
st.title("Ravi Chandera")

# intro
st.write("Hello there, I am pursuing masters in Artificial Intelligence at IIT patna. I have a strong foundation in advanced statistical analysis, machine learning, and deep learning techniques, enabling me to develop accurate and robust predictive models.")
st.write("My working includes data preprocessing, feature engineering, and dimensionality reduction techniques to extract meaningful insights and improve model performance.")
st.write("My programming background in Python helps to utilize libraries such as NumPy, Pandas, and scikit-learn to implement data science algorithms and perform data manipulation and visualization.")
st.write("specialization in natural language processing (NLP) helps me to understand both traditional and state-of-the-art NLP techniques, allowing me to design and develop innovative solutions for language-related challenges")
st.write("I know design and implementation of convolutional neural networks (CNNs) and advanced architectures such as ResNet, VGG, and Inception, allowing me to achieve state-of-the-art performance on image classification, object detection, and semantic segmentation tasks.")

# course information
df = pd.DataFrame({
    "Courses taken" : ["Machine learning","Artificial intelligence","Deep learning","Natural language processing","Big Data","statistics and probability","Optimisation","Cyber security with Blockchain","Industry 4.0"]
    })

st.dataframe(df)


st.markdown("""
## ***"A picture is worth more than a thousand words"***
""")





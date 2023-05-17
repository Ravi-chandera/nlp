import streamlit as st
from transformers import pipeline
from urllib.error import URLError
from PIL import Image


def intro():
    import streamlit as st

    st.write("# Welcome to Text Summarizer 👋")
    st.write("This is remarkable text summarizer! Unlock the power of condensing lengthy texts into concise summaries with just a few clicks. Effortlessly digest complex information as our tool analyzes and distills key points, saving you valuable time and effort. Impressively accurate and efficient, our text summarizer is your go-to companion for extracting essential insights from any document. Experience the convenience of quick summaries that enable you to grasp the essence of texts without any prior knowledge.")
    st.sidebar.success("Select a demo above.")
    summarizer = pipeline("summarization")
    userInputText = st.text_input("enetr text that you want to summarise")
    summary = summarizer(userInputText, max_length=1000, min_length=150, do_sample=False)
    st.write(summary[0]['summary_text'])




   

# def mapping_demo():
#     from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
#     import torch
#     try:
#         model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#         feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#         tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#         max_length = 16
#         num_beams = 4
#         gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
#         def predict_step(image_paths):
#             images = []
#             for image_path in image_paths:
#                 i_image = Image.open(image_path)
#                 if i_image.mode != "RGB":
#                     i_image = i_image.convert(mode="RGB")

#                 images.append(i_image)

#             pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#             pixel_values = pixel_values.to(device)

#             output_ids = model.generate(pixel_values, **gen_kwargs)

#             preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#             preds = [pred.strip() for pred in preds]
#         return preds

        
#     except URLError as e:
#         st.error(
#             """
#             **This demo requires internet access.**

#             Connection error: %s
#         """
#             % e.reason
#         )
#     import streamlit as st

# # Title or introduction for the app
#     st.title("Image Uploader")

# # Create a file uploader widget
#     uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# # Check if the user has uploaded a file
#     if uploaded_file is not None:
#     # Display the uploaded image
#         st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
#     # Perform further processing on the uploaded image (e.g., image classification, object detection, etc.)
#     # Add your own code here to process the image
    
#     # Example: You can save the uploaded image to a file
#         with open("uploaded_image.jpg", "wb") as f:
#             f.write(uploaded_file.getbuffer())
#             st.success("Image saved successfully!")
#     output = predict_step([uploaded_file]) 
#     st.write(output)


# def plotting_demo():
#     import streamlit as st
#     import time
#     import numpy as np

#     st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
#     st.write(
#         """
#         This demo illustrates a combination of plotting and animation with
# Streamlit. We're generating a bunch of random numbers in a loop for around
# 5 seconds. Enjoy!
# """
#     )

#     progress_bar = st.sidebar.progress(0)
#     status_text = st.sidebar.empty()
#     last_rows = np.random.randn(1, 1)
#     chart = st.line_chart(last_rows)

#     for i in range(1, 101):
#         new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#         status_text.text("%i%% Complete" % i)
#         chart.add_rows(new_rows)
#         progress_bar.progress(i)
#         last_rows = new_rows
#         time.sleep(0.05)

#     progress_bar.empty()

#     # Streamlit widgets automatically run the script from top to bottom. Since
#     # this button is not connected to any other logic, it just causes a plain
#     # rerun.
#     st.button("Re-run")


# def data_frame_demo():
#     import streamlit as st
#     import pandas as pd
#     import altair as alt

#     from urllib.error import URLError

#     st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
#     st.write(
#         """
#         This demo shows how to use `st.write` to visualize Pandas DataFrames.

# (Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
# """
#     )

#     @st.cache_data
#     def get_UN_data():
#         AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
#         df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
#         return df.set_index("Region")

#     try:
#         df = get_UN_data()
#         countries = st.multiselect(
#             "Choose countries", list(df.index), ["China", "United States of America"]
#         )
#         if not countries:
#             st.error("Please select at least one country.")
#         else:
#             data = df.loc[countries]
#             data /= 1000000.0
#             st.write("### Gross Agricultural Production ($B)", data.sort_index())

#             data = data.T.reset_index()
#             data = pd.melt(data, id_vars=["index"]).rename(
#                 columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
#             )
#             chart = (
#                 alt.Chart(data)
#                 .mark_area(opacity=0.3)
#                 .encode(
#                     x="year:T",
#                     y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
#                     color="Region:N",
#                 )
#             )
#             st.altair_chart(chart, use_container_width=True)
#     except URLError as e:
#         st.error(
#             """
#             **This demo requires internet access.**

#             Connection error: %s
#         """
#             % e.reason
#         )

page_names_to_funcs = {
    "—": intro,
    # "Plotting Demo": plotting_demo,
#     "Mapping Demo": mapping_demo,
    # "DataFrame Demo": data_frame_demo
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

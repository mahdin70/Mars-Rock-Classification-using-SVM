import streamlit as st
from PIL import Image
import numpy as np
import pickle
import cv2
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title='Mars Rock Classification', page_icon='🪨')
st.title('Project Altair : Mars Rock Image Classification')

model = pickle.load(open('img_model.pkl', 'rb'))

def classify_image(image):
    img_resized = cv2.resize(np.array(image), (150, 150))
    flat_data = img_resized.flatten()
    flat_data = np.array([flat_data])
    y_out = model.predict(flat_data)
    y_prob = model.predict_proba(flat_data)
    return y_out[0], y_prob[0] * 100  # Return probabilities for all classes

# Uploading Image Code
uploaded_file = st.file_uploader("Upload the Rock Image (Keep this in Rocks)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict Rock Type'):
        predicted_class, probabilities = classify_image(image)
        categories = ['Basalt', 'Coal', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Sandstone']

        sorted_probs_indices = np.argsort(probabilities)[::-1]

        chart_data = {categories[i]: probabilities[i] for i in sorted_probs_indices}
        chart_df = pd.DataFrame.from_dict(chart_data, orient='index', columns=['Matching Percentage'])

        chart_df.index.name = 'Rock Type'

        st.markdown('<div class="stDataFrame">', unsafe_allow_html=True)
        st.dataframe(chart_df.style.bar(subset=['Matching Percentage'], color='#3783bb'))
        st.markdown('</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots()
        ax.bar(chart_df.index, chart_df['Matching Percentage'], color='#3783bb')
        ax.set_ylabel('Matching Percentage')
        ax.set_xlabel('Rock Type')
        ax.set_ylim([0, 100])
        st.markdown('<div class="stBarChart">', unsafe_allow_html=True)
        st.pyplot(fig, clear_figure=True)
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

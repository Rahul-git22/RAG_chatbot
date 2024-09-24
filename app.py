import os
import streamlit as st
from app_utility import get_answer

# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Configure the Streamlit app
st.set_page_config(
    page_title="Chat with Doc",
    page_icon="0",
    layout="centered"
)

# Display title
st.title('Document Q&A LLAMA 3.1')

# File upload for PDF documents
uploaded_file = st.file_uploader(label='Upload your file',type = ["pdf"])

# Text input for user question
user_query = st.text_input('Ask your Question')

# Run button to trigger analysis
if st.button("Run"):
    bytes_data = uploaded_file.read()
    file_name = uploaded_file.name
    file_path = os.path.join(working_dir,file_name)
    with open(file_path,"wb") as f:
        f.write(bytes_data)
    answer= get_answer(file_name, user_query)

    st.success(answer)

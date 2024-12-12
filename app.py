import streamlit as st
import ollama
from PIL import Image
from streamlit_navigation_bar import st_navbar

import contsants as CNT

st.set_page_config(
    page_title="OCR2LaTeX",
    page_icon="🔍",
    menu_items={
        'Get Help': 'https://github.com/bharath03-a/OCR2LaTeX',
        'Report a bug': "https://github.com/bharath03-a/OCR2LaTeX",
        'About': "**OCR2LaTeX** is a Streamlit-based application that converts images of mathematical formulas or tables into LaTeX code. Using OCR and a fine-tuned LLaMA model, \
                    the app provides LaTeX code along with usage instructions, making it easier to integrate into your documents."
    }
)

st.title("OCR2LaTeX using llama3.2 🦙")

def clear_chat():
    if "messages" in st.session_state:
        del st.session_state["messages"]
    st.rerun()

with st.sidebar:
    st.title("OCR2LaTeX using llama3.2")
    st.write("OCR2LaTeX is a Streamlit-based application that converts images of mathematical formulas or tables into LaTeX code.")
    
    uploaded_file = st.file_uploader("Choose a File: ",
                                type = ['png', 'jpg', 'jpeg'],
                                help = "Please upload the screenshot or image of the Latex formula")
    
    col1, col2 = st.columns([0.5, 0.5])
    if uploaded_file is not None:
        uploaded_img = Image.open(uploaded_file)
        st.image(uploaded_img, caption = "Uploaded image of the Mathematical Formula")
        with col1:
            if st.button("Extract LateX", icon = "🕵️‍♂️", type = "primary"):
                try:
                    with st.spinner("Processing the image, please wait..."):
                        model_response = ollama.chat(
                            model = "llama3.2-vision",
                            messages = [{
                                "role" : "user",
                                "content" : CNT.DEFAULT_MESSAGE_CONTENT,
                                "images" : [uploaded_file.getvalue()]
                            }]
                        )
                        st.session_state['messages'] = model_response.message.content
                except Exception as e:
                    st.error(f"An error has interrupted the processing of the file: [{e}]")

    with col2:
        if st.button("Clear Chat", icon = "🗑️", help = "Clears your previous chat"):
            clear_chat()

if "messages" in st.session_state:
    st.write("#### LaTeX code from the Model:")
    st.code(st.session_state['messages'], language = "latex", line_numbers = True)

    st.write("#### Rendered LaTeX code from the Model:")
    cleaned_latex = st.session_state['messages'].replace(r"\[", "").replace(r"\]", "")
    st.latex(cleaned_latex)
else:
    st.info("Upload an image and click 'Extract LaTeX' to see the results here.")
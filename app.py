import streamlit as st
import ollama
from PIL import Image
from streamlit_navigation_bar import st_navbar

st.set_page_config(
    page_title="OCR2LateX",
    page_icon="🔍",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/bharath03-a/OCR2LaTeX',
        'Report a bug': "https://github.com/bharath03-a/OCR2LaTeX",
        'About': "**OCR2LaTeX** is a Streamlit-based application that converts images of mathematical formulas or tables into LaTeX code. Using OCR and a fine-tuned LLaMA model, \
                    the app provides LaTeX code along with usage instructions, making it easier to integrate into your documents."
    }
)

st.title("OCR2LateX using llama3.2 🦙")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = ollama.chat(
            model="llama3.2-vision",
            messages=[{"role": m["role"],
                       "content": m["content"]}
                for m in st.session_state.messages
            ],
        )

        assistant_response = response.message.content

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        st.error(f"Error: {e}")
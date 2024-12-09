import streamlit as st
import ollama
from PIL import Image

st.title("OCR2LateX Generator using llama3.2 🦙")


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
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
        )
        st.write("Full Response:", response)

        if "content" in response:
            assistant_response = response["content"]
        else:
            st.error("Unexpected response structure, 'content' key not found.")
            assistant_response = "No valid response from the model."

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        st.error(f"Error: {e}")
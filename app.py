# app.py
import streamlit as st
import ollama
from PIL import Image
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from streamlit_chat import message
from io import BytesIO

from langchain_ollama.llms import OllamaLLM

# --- Set Page Config ---
st.set_page_config(
    page_title="OCR2LaTeX",
    page_icon="🔍",
    menu_items={
        'Get Help': 'https://github.com/bharath03-a/OCR2LaTeX',
        'Report a bug': "https://github.com/bharath03-a/OCR2LaTeX",
        'About': "OCR2LaTeX is a Streamlit-based application that converts images of mathematical formulas or tables into LaTeX code."
    }
)

# --- Title ---
st.title("OCR2LaTeX using LLaMA3.2 🦙")

# --- Sidebar for Uploading Multiple Files ---
with st.sidebar:
    st.title("Upload Files")
    uploaded_files = st.file_uploader("Choose Files", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

# --- RAG Setup ---
@st.cache_resource
def load_knowledge_base():
    """
    Load the Chroma knowledge base using HuggingFace embeddings.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="latex_knowledge_base",  # Directory where the Chroma DB is stored
        embedding_function=embedding_model
    )
    return vectorstore

knowledge_base = load_knowledge_base()

@st.cache_resource
def initialize_llm():
    """
    Initialize the Ollama LLM.
    """
    return OllamaLLM(model="llama2", base_url="http://localhost:11434")  # Adjust the model and base URL as needed

llm = initialize_llm()

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=knowledge_base.as_retriever())

# --- Process Uploaded Files ---
def process_image(file):
    """
    Process an uploaded image to extract LaTeX code using the LLaMA model.
    """
    uploaded_img = Image.open(file)
    st.image(uploaded_img, caption=f"Processing {file.name}")
    try:
        with st.spinner("Processing image..."):
            response = llm.chat(
                messages=[{"role": "user", "content": "Extract LaTeX", "images": [file.getvalue()]}]
            )
            return response["message"]["content"]
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- Download LaTeX File ---
def download_latex(latex_code, file_name="output.tex"):
    """
    Provide a download button for the extracted LaTeX code.
    """
    tex_file = BytesIO(latex_code.encode())
    st.download_button(
        label="Download LaTeX File",
        data=tex_file,
        file_name=file_name,
        mime="text/plain"
    )

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    message(msg["content"], is_user=msg["role"] == "user")

user_input = st.text_input("Ask something about the output:")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    response = qa_chain.run(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    message(response)

# --- Main Logic ---
if uploaded_files:
    for file in uploaded_files:
        latex_code = process_image(file)
        if latex_code:
            st.write("### Extracted LaTeX Code:")
            st.code(latex_code, language="latex")
            st.write("### Rendered LaTeX:")
            cleaned_latex = latex_code.replace(r"\[", "").replace(r"\]", "")
            st.latex(cleaned_latex)
            download_latex(latex_code, file.name.replace(".png", ".tex").replace(".jpg", ".tex"))
else:
    st.info("Upload images to extract LaTeX.")
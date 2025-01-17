import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class DocumentProcessor:
    """
    Attributes: 
        directory (str): The directory containing the documents to load.
    """
    def __init__(self, directory):
        """
        Initializes the DocumentProcessor with the given directory.
        Args:
            directory (str): Path to the directory containing documents.
        """
        self.directory = directory

    def load_documents(self):
        """
        Loads all documents from the specified directory. Supports `.txt` and `.pdf` files.
        Returns:
            list: A list of documents loaded from the directory.
        """
        documents = []
        for file in os.listdir(self.directory):
            file_path = os.path.join(self.directory, file)
            if file.endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        return documents

class TextSplitter:
    """
    Attributes:
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap size between consecutive chunks.
    """
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        """
        Initializes the TextSplitter with specified chunk size and overlap.
        Args:
            chunk_size (int): Maximum size of each chunk. Default is 1000.
            chunk_overlap (int): Overlap size between consecutive chunks. Default is 100.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents):
        """
        Splits a list of documents into smaller chunks.
        Args:
            documents (list): A list of documents to split.
        Returns:
            list: A list of split documents.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(documents)

class VectorStoreManager:
    """
    Attributes:
        persist_directory (str): Directory to store the Chroma database.
        embedding_model (HuggingFaceEmbeddings): Embedding model for vector generation.
    """
    def __init__(self, persist_directory, embedding_model_name):
        """
        Initializes the VectorStoreManager with the given persistence directory and embedding model.
        Args:
            persist_directory (str): Path to the directory for storing the Chroma database.
            embedding_model_name (str): Name of the embedding model.
        """
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    def create_vectorstore(self, documents):
        """
        Creates and saves a Chroma vector store from the provided documents.
        Args:
            documents (list): A list of split documents.
        Returns:
            Chroma: The created Chroma vector store.
        """
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        print(f"{vectorstore._collection.count()} vectors stored in Chroma vector store.")
        print(f"Chroma vector store saved locally in '{self.persist_directory}'.")
        return vectorstore

class KnowledgeBaseBuilder:
    """
    Attributes:
        knowledge_base_directory (str): Path to the directory containing knowledge base documents.
        persist_directory (str): Directory to save the Chroma vector store.
        embedding_model_name (str): Name of the embedding model.
    """
    def __init__(self, knowledge_base_directory, persist_directory, embedding_model_name):
        """
        Initializes the KnowledgeBaseBuilder with the required directories and model name.
        Args:
            knowledge_base_directory (str): Path to the knowledge base directory.
            persist_directory (str): Path to the directory for saving the vector store.
            embedding_model_name (str): Name of the embedding model.
        """
        self.knowledge_base_directory = knowledge_base_directory
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name

    def build_vectorstore(self):
        """
        Builds the Chroma vector store by loading documents, splitting them,
        and storing the embeddings in the specified directory.
        """
        # Step 1: Load documents
        processor = DocumentProcessor(self.knowledge_base_directory)
        documents = processor.load_documents()

        # Step 2: Split documents
        splitter = TextSplitter()
        split_docs = splitter.split_documents(documents)
        print(f"Loaded and split {len(split_docs)} documents.")

        # Step 3: Create vector store
        vector_manager = VectorStoreManager(self.persist_directory, self.embedding_model_name)
        vector_manager.create_vectorstore(split_docs)

if __name__ == "__main__":
    # initializing and run the knowledge base builder
    builder = KnowledgeBaseBuilder(
        knowledge_base_directory="knowledge_base",
        persist_directory="latex_knowledge_base",
        embedding_model_name='all-MiniLM-L6-v2'
    )
    builder.build_vectorstore()
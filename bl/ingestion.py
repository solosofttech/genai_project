import os
import tempfile
import chromadb
from bl import *
import streamlit as st
from typing import List, Optional
from chromadb.config import Settings
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Ingestion:    
    
    @staticmethod
    def get_embeddings_model():
        
        try:
            embeddings = OllamaEmbeddings(model=G_EMBEDDING_MODEL,base_url=G_BASE_URL)
            return embeddings
        except Exception as e:
            st.error(f"Error initializing embeddings model: {str(e)}")
            st.error(f"Please make sure Ollama is running and the mxbai-embed-large model is available")
            return None

    @staticmethod
    def split_text(text: str) -> List[Document]:
        """Split text into chunks"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size= G_CHUNK_SIZE,
                chunk_overlap= G_CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_text(text)
            documents = [Document(page_content=chunk, metadata={"source": "uploaded_document"}) for chunk in chunks]
            
            return documents
            
        except Exception as e:
            st.error(f"Error splitting text: {str(e)}")
            return []

    @staticmethod
    def create_vectorstore(documents: List[Document], embeddings) -> Optional[Chroma]:
        """Create and return ChromaDB vector store"""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(G_PERSIST_DIRECTORY, exist_ok=True)
            
            # Initialize ChromaDB with persistent storage
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=G_PERSIST_DIRECTORY,
                collection_name=G_CHROMA_COLLECTION_NAME
            )
            
            # Persist the vector store
            vectorstore.persist()
            
            return vectorstore
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None
        
    @staticmethod
    def load_document(uploaded_file):
        try:
            # create a temporary file to save uploaded file
            with tempfile.NamedTemporaryFile (delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # get file extension
            # Determine file type and initialize loader 
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension == 'txt':
                loader = TextLoader(tmp_file_path, encoding='utf-8')
            elif file_extension in ['docx', 'doc']:
                loader = Docx2txtLoader(tmp_file_path)
            else:
                # Try unstructured loader for other file types
                loader = UnstructuredFileLoader(tmp_file_path)

            documents = loader.load()
            text = "\n".join([doc.page_content for doc in documents])
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return text
            
        except Exception as e:
            st.write(f"❌ Error loading document {e}")            
            return None


    @staticmethod
    def ingestion():        
        st.title("Document Ingestion Pipeline")
        st.write("Upload your document to create embeddings and store them in ChromaDB")

        uploaded_file = st.file_uploader("Choose a file", type=["docx","doc", "txt", "pdf"])

        if uploaded_file is not None:
            # Display file info
            st.write(f"**File uploaded:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size} bytes")        

            #Step 1: initialize embeddings
            with st.spinner("Initializing embeddings model..."):
                embeddings = Ingestion.get_embeddings_model()

            if embeddings is None:
                st.write("❌ Error Initializing Embedding Model")
                st.stop()
            
            #Step 3: get uploaded document text
            with st.spinner("Getting text from the uploaded file"):
                text = Ingestion.load_document(uploaded_file)

            if text is None:
                st.write("❌ Error File is empty. No text found")
                st.stop()

            st.success(f"✅ Document loaded successfully! Text length: {len(text)} characters")

            # Step 4: split text into chunks
            documents = Ingestion.split_text(text)
            
            if not documents:
                st.error("Failed to split text into chunks")
                st.stop()
                    
            st.success(f"✅ Text split into {len(documents)} chunks")
            
            # Display sample chunks
            with st.expander("View sample chunks"):
                for i, doc in enumerate(documents[:3]):
                    st.write(f"**Chunk {i+1}:**")
                    st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
            
            # Step 3: Create embeddings and store in ChromaDB
            st.write("🔍 **Step 3:** Creating embeddings and storing in ChromaDB...")
            vectorstore = Ingestion.create_vectorstore(documents, embeddings)
            
            if vectorstore is None:
                st.error("Failed to create vector store")
                st.stop()
            
            st.success("✅ Document successfully processed and stor ed in ChromaDB!")
            
            # Display statistics
            st.write("📊 **Processing Summary:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Chunks", len(documents))
            
            with col2:
                st.metric("Chunk Size", f"{G_CHUNK_SIZE} chars")
            
            with col3:
                st.metric("Chunk Overlap", f"{G_CHUNK_OVERLAP} chars")
            
            # Display vector store info
            st.write("🗄️ **Vector Store Information:**")
            st.write(f"- **Storage Location:** {G_PERSIST_DIRECTORY}")
            st.write(f"- **Embedding Model:** {G_EMBEDDING_MODEL}")            
            st.write(f"- **Collection Name:** documents")
            
            st.balloons()









        
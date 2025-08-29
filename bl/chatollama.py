import os
from bl import *
import chromadb
import streamlit as st
from typing import List, Optional
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class ChatOllama:

    @staticmethod
    def initialize_models():
        """Initialize LLM and embeddings models"""
        try:
            # Initialize LLM
            llm = OllamaLLM(
                model=G_LLM_LAMA_MODEL,
                base_url=G_BASE_URL,
                temperature=0.7
            )
            
            # Initialize embeddings model
            embeddings = OllamaEmbeddings(
                model=G_EMBEDDING_MODEL,
                base_url=G_BASE_URL
            )
            
            return llm, embeddings
            
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            st.error("Please make sure Ollama is running and the required models are available")
            return None, None
        
    @staticmethod
    def load_vectorstore(embeddings):
        """Load existing ChromaDB vector store"""
        try:
            if not os.path.exists(G_PERSIST_DIRECTORY):
                st.warning("No vector store found. Please upload documents first using the ingestion pipeline.")
                return None
            
            vectorstore = Chroma(
                persist_directory=G_PERSIST_DIRECTORY,
                embedding_function=embeddings,
                collection_name=G_CHROMA_COLLECTION_NAME
            )
            
            # Check if the collection has documents
            collection = vectorstore._collection
            if collection.count() == 0:
                st.warning("Vector store is empty. Please upload documents first using the ingestion pipeline.")
                return None
            
            return vectorstore
            
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return None
    
    @staticmethod
    def get_relevant_documents(vectorstore, query: str, k: int = 3):
        """Get relevant documents from vector store"""
        try:
            docs = vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
    @staticmethod
    def format_context(documents: List[Document]) -> str:
        """Format retrieved documents as context"""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            if len(content) > 500:
                content = content[:500] + "..."
            context_parts.append(f"Document {i}:\n{content}")
        
        return "\n\n".join(context_parts)
    
    @staticmethod
    def generate_response(llm, query: str, context: str = "") -> str:
        """Generate response using LLM"""
        try:
            if context:
                prompt = f"""Based on the following context, please answer the user's question. If the context doesn't contain relevant information to answer the question, respond with "Sorry, answer not found from local vector store."

                Context:
                {context}

                Question: {query}

                Answer:"""
            else:
                prompt = f"""Please answer the following question: {query}"""
            
            response = llm.invoke(prompt)
            return response.strip()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while generating the response."

    @staticmethod
    def chat_interface():
        """Main chat interface"""
        st.title("🤖 ChatLama - Local AI Chat")
        st.write("Ask questions about your uploaded documents using the local Gemma3:1b model")
        
        # step 1: Initialize embeddings and llama models
        with st.spinner("Initializing models..."):
            llm, embeddings = ChatOllama.initialize_models()

        if llm is None or embeddings is None:
            st.stop()

        # step2: Load vector store
        vectorstore = ChatOllama.load_vectorstore(embeddings)
        
        if vectorstore is None:
            st.info("💡 **Tip:** Use the ingestion pipeline to upload documents before chatting!")
            st.stop()

         # Display vector store info
        collection = vectorstore._collection
        st.sidebar.write("📊 **Vector Store Info:**")
        st.sidebar.write(f"- Documents: {collection.count()}")
        st.sidebar.write(f"- Model: {G_LLM_LAMA_MODEL}")
        st.sidebar.write(f"- Embeddings: {G_EMBEDDING_MODEL}")

        # step3: setup chat history    
        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

         # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get relevant documents
            with st.spinner("Searching for relevant information..."):
                relevant_docs = ChatOllama.get_relevant_documents(vectorstore, prompt)
            
            if relevant_docs is None:
                st.error("Relevant document not found")
                st.stop()            

            # Generate response
            with st.spinner("Generating response..."):
                if relevant_docs:
                    context = ChatOllama.format_context(relevant_docs)                          
                    response = ChatOllama.generate_response(llm, prompt, context)                                        

                    # Check if response indicates no relevant info found
                    if "sorry, answer not found from local vector store" in response.lower():
                        response = "Sorry, answer not found from local vector store."
                else:
                    response = "Sorry, answer not found from local vector store."

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Display context if available
            if relevant_docs and "sorry, answer not found from local vector store" not in response.lower():
                with st.expander("🔍 View relevant context used"):
                    for i, doc in enumerate(relevant_docs, 1):
                        st.write(f"**Document {i}:**")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.divider()

            # Sidebar controls
        with st.sidebar:
            st.write("⚙️ **Settings:**")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
            
            # Model info
            st.write("---")
            st.write("📋 **Model Information:**")
            st.write(f"- **LLM:** {G_LLM_LAMA_MODEL}")
            st.write(f"- **Embeddings:** {G_EMBEDDING_MODEL}")
            st.write(f"- **Vector Store:** ChromaDB")
            
            # Instructions
            st.write("---")
            st.write("💡 **How to use:**")
            st.write("1. Upload documents using the ingestion pipeline")
            ##### 2. Ask questions about your documents
            ##### 3. The AI will search through your documents and provide relevant answers")
            
            # Requirements
            st.write("---")
            st.write("🔧 **Requirements:**")
            st.write("- Ollama running on localhost:11434")
            st.write("- gemma3:1b model installed")
            st.write("- mxbai-embed-large model installed")
            
            # Install commands
            with st.expander("📥 Install Models"):
                st.code("""
                # Install required models in Ollama:
                ollama pull gemma3:1b
                ollama pull mxbai-embed-large
                """)



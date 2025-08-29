import os
import wave
import requests
from bl import *
import numpy as np
import streamlit as st
from typing import List, Optional, Dict, Any
from bl.ingestion import Ingestion
from langchain_groq import ChatGroq
from elevenlabs.client import ElevenLabs
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False
    audio_recorder = None

# Initializing Embedding Model            
embeddings = HuggingFaceEmbeddings(model_name=G_GROQ_EMBED_MODEL)

# Initializing LLM            
llm = ChatGroq(
    temperature=0,
    model_name=G_GROQ_LLM_MODEL,
)

class ChatGroqModel: 

    @staticmethod
    def initialize_models():
        """Initialize LLM and embeddings models"""
        try:
            # Initializing Embedding Model            
            embeddings = HuggingFaceEmbeddings(model_name=G_GROQ_EMBED_MODEL)

            # Initializing LLM            
            llm = ChatGroq(
                temperature=0,
                model_name=G_GROQ_LLM_MODEL,
            )

            return llm, embeddings
            
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")            
            return None, None
        
    @staticmethod
    def transcribe_audio_with_elevenlabs(audio_bytes: bytes, mime_type: str = "audio/wav") -> Optional[str]:
        """Send audio bytes to ElevenLabs STT and return transcribed text.

        Tries official SDK first, falls back to REST if needed.
        """
        if not G_GROQ_API_KEY:
            st.error("ELEVEN_API_KEY not set in .env")
            return None

        # Try SDK
        try:
            client = ElevenLabs(api_key=G_ELEVEN_API_KEY)
            # Some SDK versions expose speech_to_text.transcribe; handle gracefully
            if hasattr(client, "speech_to_text") and hasattr(client.speech_to_text, "transcribe"):
                response = client.speech_to_text.transcribe(audio=audio_bytes, model_id="scribe_v1")
                # Response may be dict-like with 'text'
                text = response.get("text") if isinstance(response, dict) else None
                if text:
                    return text
            # If method not available or returned nothing, fall through to REST
        except Exception as _:
            pass

        # Fallback to REST API
        try:
            headers = {"xi-api-key": G_ELEVEN_API_KEY}
            files = {
                "file": ("audio.wav", audio_bytes, mime_type),
            }
            data = {"model_id": "scribe_v1"}
            resp = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers=headers,
                files=files,
                data=data,
                timeout=60,
            )
            if resp.ok:
                payload = resp.json()
                # Common key is 'text' for transcript
                return payload.get("text") or payload.get("transcript")
            else:
                st.error(f"Transcription failed: {resp.status_code} {resp.text}")
                return None
        except Exception as ex:
            st.error(f"Transcription error: {str(ex)}")
            return None
    
    @staticmethod
    def search_web(query: str) -> Dict[str, Any]:
        """Search the web using Serper API and return relevant results."""
        if not G_SERPER_API_KEY:
            return {"error": "SERPER_API_KEY not configured"}
        
        try:
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": G_SERPER_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "q": query,
                "num": 5  # Number of results
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Search failed: {response.status_code}"}
        except Exception as e:
            return {"error": f"Search error: {str(e)}"}
        
    @staticmethod
    def extract_web_content(search_results: Dict[str, Any]) -> str:
        """Extract and format web search content for LLM consumption."""
        if "error" in search_results:
            return f"Web search failed: {search_results['error']}"
        
        content_parts = []
        
        # Extract organic results
        if "organic" in search_results:
            for i, result in enumerate(search_results["organic"][:3], 1):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                content_parts.append(f"Result {i}: {title}\n{snippet}")
        
        # Extract knowledge graph if available
        if "knowledgeGraph" in search_results:
            kg = search_results["knowledgeGraph"]
            title = kg.get("title", "")
            description = kg.get("description", "")
            if title and description:
                content_parts.append(f"Knowledge: {title}\n{description}")
        
        return "\n\n".join(content_parts) if content_parts else "No relevant web results found."
        
    @staticmethod
    def chatbot():
        try:
            """Main chat interface"""
            st.title("🤖 Chat Using Groq ")
            st.write(f"Ask questions about your uploaded documents using the {G_GROQ_LLM_MODEL} model")

            uploaded_file = st.file_uploader("Choose a file", type=["docx","doc", "txt", "pdf"])
            
            if uploaded_file:

                text = Ingestion.load_document (uploaded_file)  
                
                if text is None:
                    st.error("❌ Error File is empty. No text found")
                    st.stop()

                st.success(f"✅ Document loaded successfully! Text length: {len(text)} characters")  

                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=500,
                    chunk_overlap=50,
                )

                splitted_text = text_splitter.split_text(text)
                
                if not splitted_text:
                    st.error("Failed to split text into chunks")
                    st.stop()
                        
                st.success(f"✅ Text split into {len(splitted_text)} chunks")

                # step 5: Initialize embeddings and llama models
                #with st.spinner("Initializing models..."):
                #    llm, embeddings = ChatGroqModel.initialize_models()

                if llm is None or embeddings is None:
                    st.error(f"Failed to initialize the llm and embedding model")
                    st.stop()

                # step6: create embeddgins
                document = FAISS.from_texts(splitted_text, embedding=embeddings)

                # step7: initialize qa chain
                qa = load_qa_chain(llm=llm, chain_type="stuff")    

                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Input mode selector
                input_mode = st.radio(
                    "Choose input method",
                    ("Type", "Voice"),
                    horizontal=True,
                )

                # Display chat
                for message in st.session_state["messages"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                user_prompt = None

                if input_mode == "Type":
                    user_prompt = st.chat_input("Say something")
                else:
                    st.write("Click to record your question and release to stop.")
                    audio_bytes = audio_recorder(text="Record", icon_size="2x")
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
                        with st.spinner("Transcribing audio..."):                           
                            transcript = ChatGroqModel.transcribe_audio_with_elevenlabs(audio_bytes, mime_type="audio/wav")
                        if transcript:
                            st.success("Transcription complete")
                            user_prompt = transcript
                        else:
                            st.warning("Could not transcribe audio. Please try again or type your question.")   

                if user_prompt:

                    st.chat_message("user").markdown(user_prompt)

                    st.session_state.messages.append(
                        {"role": "user", "content": user_prompt}
                    )

                    retriver = document.similarity_search(user_prompt)
                    context_text = " ".join([doc.page_content for doc in retriver])

                    # Generate response
                    with st.spinner("Generating response..."):
                        if retriver:
                            llm_response = qa.invoke( {"input_documents": retriver, "question": user_prompt})
                            # Check if the answer is found in the local context
                            response = llm_response["output_text"]      

                            if any(phrase in response.lower() for phrase in ["i don't know", "not found", "cannot answer", "no information"]):
                                # Check if response indicates no relevant info found                           
                                response = "Sorry, answer not found from local vector store."
                        else:
                            response = "Sorry, answer not found from local vector store."

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    
                    # Display context if available
                    if retriver and "sorry, answer not found from local vector store" not in response.lower():
                        with st.expander("🔍 View relevant context used"):
                            for i, doc in enumerate(retriver, 1):
                                st.write(f"**Document {i}:**")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                st.divider()
                    else:
                        # search web using function calling                        
                        with st.spinner("Searching the web for additional information..."):
                            web_results = ChatGroqModel.search_web(user_prompt)
                            web_content = ChatGroqModel.extract_web_content(web_results)

                        if "error" not in web_results:
                            # Create a comprehensive response using both local and web data
                            combined_context = f"Local Documents:\n{context_text}\n\nWeb Information:\n{web_content}"
                            
                            # Use LLM to generate a comprehensive answer
                            comprehensive_prompt = f"""Based on the following information, provide a comprehensive answer to the user's question. 
                            If the local documents don't contain the answer, rely on the web information.
                            
                            Local Documents Context:
                            {context_text}
                            
                            Web Information:
                            {web_content}
                            
                            User Question: {user_prompt}
                            
                            Please provide a detailed answer combining both sources when possible."""
                            
                            comprehensive_response = llm.invoke(comprehensive_prompt)
                            final_response = comprehensive_response.content
                            response_source = "web_search"
                        else:
                            # Web search failed, provide local answer with disclaimer
                            final_response = f"{response}\n\nNote: I couldn't search the web for additional information due to an error: {web_results['error']}"
                            response_source = "local_only"

                        with st.chat_message("assistant"):
                            st.markdown(final_response)
                            
                            # Show source indicator
                            if response_source == "web_search":
                                st.info("🔍 Answer enhanced with web search results")
                            elif response_source == "local_only":
                                st.warning("⚠️ Answer from local documents only (web search unavailable)")

                        st.session_state.messages.append(
                            {"role": "assistant", "content": final_response}
                        )

        except Exception as ex:
            st.error(f"Exception:{str(ex)}")
            st.stop()
    
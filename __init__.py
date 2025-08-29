import os
from dotenv import load_dotenv

load_dotenv()

# load ollama global variables
G_BASE_URL = os.getenv("base_url")
G_LLM_LAMA_MODEL = os.getenv("llm_model") 
G_CHUNK_SIZE  = int(os.getenv("chunk_size"))
G_EMBEDDING_MODEL = os.getenv("embeddings_model")
G_CHUNK_OVERLAP = int (os.getenv("chunk_overlap") )
G_PERSIST_DIRECTORY = os.getenv("persist_directory")
G_CHROMA_COLLECTION_NAME = os.getenv("collection_name") 

# load groq global variables
G_GROQ_API_KEY=  os.getenv("groq_api_key")
G_GROQ_EMBED_MODEL= os.getenv("groq_embed_model_name")
G_GROQ_LLM_MODEL= os.getenv("groq_model_name")
G_ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
G_SERPER_API_KEY=os.getenv("SERPER_API_KEY")

__all__ = ["G_EMBEDDING_MODEL",
           "G_CHUNK_SIZE",
           "G_CHUNK_OVERLAP",
           "G_PERSIST_DIRECTORY",
           "G_BASE_URL", 
           "G_LLM_LAMA_MODEL",
           "G_CHROMA_COLLECTION_NAME",
           "G_GROQ_API_KEY",
           "G_GROQ_EMBED_MODEL",
           "G_GROQ_LLM_MODEL",
           "G_ELEVEN_API_KEY",
           "G_SERPER_API_KEY"]
    
import os
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores.base import VectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore

class RagService:
    def __init__(self, api_key:str, local_data: str, local_vector: str):
        self.__api_key = api_key
        self.__local_data = local_data
        self.__local_vector = local_vector

    def load_document(self) -> list[Document]:
        loader = PyMuPDFLoader(file_path=self.__local_data)
        spliter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 150)
        docs = loader.load()
        return spliter.split_documents(documents=docs)
    
    def get_vector_store(self) -> VectorStore:
        embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=self.__api_key)

        if not os.path.exists(self.__local_vector) or not os.listdir(self.__local_vector):
            index = faiss.IndexFlatL2(len(embedding.embed_query("assistente de suporte")))
            
            vector_store = FAISS(
                embedding_function=embedding,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={})
            
            documents = self.load_document()
            vector_store.add_documents(documents=documents)

            vector_store.save_local(self.__local_vector)

            return vector_store

        return FAISS.load_local(self.__local_vector, embeddings=embedding, allow_dangerous_deserialization=True)

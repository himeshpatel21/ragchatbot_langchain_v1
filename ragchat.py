import os
import bs4
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader  
)

#for loading documents from url
def load_from_url(url: str) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    
    # 1. Handle PDF links
    if url.lower().endswith(".pdf"):
        loader = PyPDFLoader(url)
    else:
        # 2. Add the User-Agent header to bypass blocks
        loader = WebBaseLoader(
            web_paths=(url,),
            header_template={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
            }
        )
        # Note: I removed bs_kwargs temporarily to ensure we get ALL text first. 
        # If this works, you can add the SoupStrainer back later.

    try:
        docs = loader.load()
        
        # Check if we actually got content
        if not docs or all(not d.page_content.strip() for d in docs):
            return []

        for doc in docs:
            doc.metadata["source_type"] = "pdf" if url.lower().endswith(".pdf") else "url"
            doc.metadata["display_name"] = url
        return docs
    except Exception as e:
        print(f"Error loading URL: {e}")
        return []

from pathlib import Path
#for loading documents from pdf
def load_from_pdf(file_path:str , display_name:str="") -> List[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source_type"]="pdf"
        doc.metadata["display_name"]=display_name if display_name else Path(file_path).name

    return docs

#for user pasted copied raw text
def load_from_text(text: str, source_name:str="Pasted Text") -> List[Document]:
    doc = Document(
        page_content=text, 
        metadata={
            "source": source_name,
            "source_type": "text",
            "display_name":source_name,
        })

    return [doc]

#for user uploaded text file
def load_from_text_file(file_path:str, display_name:str="") -> List[Document]:
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    for doc in docs:
        doc.metadata["source_type"]="text_file"
        doc.metadata["display_name"]=display_name if display_name else Path(file_path).name
    return docs

#for youtube urls
def load_from_youtube(url:str) -> List[Document]:
    try:
        from langchain_community.document_loaders import YoutubeLoader
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False,language=["en", "en-IN", "en-US"])
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_type"]="youtube"
            doc.metadata["display_name"]=url
        return docs
    except Exception as e:
        raise ValueError(f"Failed to load from YouTube URL: {e}")
    
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


class ragchat:

    contextual_prompt = (
        "Given a chat history and the latest user question which might reference "
        "context in the chat history, formulate a standalone question which can be "
        "understood without the chat history. Do NOT answer the question — just "
        "reformulate it if needed, otherwise return it as is."
    )

    qa_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you don't know the answer from the context, say so honestly. "
        "Be concise (3–5 sentences max) and always cite the source when relevant.\n\n"
        "{context}"
    )

    def __init__(self, groq_api_key:str , hf_token:str =""):
        self.groq_api_key = groq_api_key
        if hf_token:
            os.environ["HF_TOKEN"]= hf_token
        
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key, 
            model="llama-3.3-70b-versatile"
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name ="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 100
        )   

        self.vectorstore: Chroma | None = None
        self.chain = None
        self.chat_history: List= []
        self.loaded_sources: List[Dict] = []
    
    def add_source(self, docs: List[Document]) -> int:

        if not docs:
            raise ValueError("No documents to add as source.")
        
        chunks = self.splitter.split_documents(docs)

        if self.vectorstore is None:
            self.vectorstore =Chroma.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)
        
        self._build_chain()

        meta = docs[0].metadata
        self.loaded_sources.append({
            "type": meta.get("source_type", "unknown"),
            "name": meta.get("display_name", meta.get("source", "unknown")),
            "chunks": len(chunks)
        })
        return len(chunks)


    def _build_chain(self):
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k":4}
        )
        conq_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextual_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, conq_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",self.qa_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        self.chain = create_retrieval_chain(history_aware_retriever,qa_chain)

    def chat(self, question:str) -> Dict[str, Any]:
        if self.chain is None:
            raise ValueError("No sources loaded yet. Please add at least one source.")
        
        response = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history
        })

        ans = response['answer']
        
        context_docs : List[Document] = response.get("context",[])

        self.chat_history.extend([
            HumanMessage(content = question),
            AIMessage(content = ans)
        ])

        seen = set()
        unique_sources = []

        for doc in context_docs:
            meta = doc.metadata
            name = meta.get("display_name") or meta.get("source","unknown")
            key = (name, meta.get("page",""))

            if key not in seen:
                seen.add(key)
                unique_sources.append({
                    "name":name,
                    "type": meta.get("source_type","unknown"),
                    "page": meta.get("page"),
                    "snippet": doc.page_content[:200].strip()
                })
        
        return {
            "answer": ans,
            "sources":unique_sources}
    

    def clear_history(self):
        self.chat_history = []

    def reset(self):
        self.vectorstore = None
        self.chain = None
        self.chat_history = []
        self.loaded_sources = []
    
    @property
    def is_ready(self) -> bool:
        return self.chain is not None
    
    @property
    def source_count(self) -> int:
        return len(self.loaded_sources) 
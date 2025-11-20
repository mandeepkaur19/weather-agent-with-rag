"""RAG (Retrieval-Augmented Generation) service for querying PDF documents."""
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from vector_store import VectorStore
from config import LLM_MODEL, OPENAI_API_KEY


class RAGService:
    """Service for RAG-based query answering from PDF documents."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize RAGService with vector store and LLM."""
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
            Use only the information from the context to answer the question. If the context doesn't contain
            enough information to answer the question, say so clearly.
            
            Context:
            {context}
            
            Question: {question}
            
            Provide a clear and concise answer based on the context above."""),
            ("human", "{question}")
        ])
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary containing answer and retrieved context
        """
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(question, top_k=top_k)
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "sources": [],
                "retrieved_chunks": []
            }
        
        # Build context from retrieved documents
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(doc["text"])
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer using LLM
        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "context": context,
            "question": question
        })
        
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "sources": [doc["metadata"] for doc in retrieved_docs],
            "retrieved_chunks": retrieved_docs
        }


#!/usr/bin/env python3
"""
Simple RAG application using LangChain v0.2 that allows querying web pages.
"""

import os
import sys
import argparse
from typing import List, Dict

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama


def load_webpage(url: str) -> List:
    """Load and parse webpage content"""
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        print("\nWebpage content:")
        print("-" * 50)
        for doc in documents:
            print(doc.page_content)
        print("-" * 50)
        return documents
    except Exception as e:
        print(f"Error loading webpage: {e}")
        return []

def split_documents(documents: List) -> List:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    print(f"\nSplit documents into {len(splits)} chunks")
    return splits

def create_vectorstore(splits: List, model: str) -> Chroma:
    """Create and populate vector store"""
    # Clear existing database to prevent dimension mismatch
    if os.path.exists("./chroma_db"):
        import shutil
        shutil.rmtree("./chroma_db")
    
    embeddings = OllamaEmbeddings(model=model)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("\nCreated vector store with Ollama embeddings")
    return vectorstore

def setup_rag_chain(vectorstore: Chroma, model: str) -> RunnablePassthrough:
    """Set up the RAG chain for querying"""
    # Initialize retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Get the default RAG prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Initialize Ollama LLM
    llm = Ollama(model=model)  # You can use other models like "mistral" or "gemma"

    # Create the RAG chain
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain

def main():
    """Main application loop"""
    parser = argparse.ArgumentParser(description="Web Page Content Loader with RAG")
    parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
    args = parser.parse_args()
    
    print(f"Web Page Content Loader using {args.model} model")
    vectorstore = None
    rag_chain = None
    
    while True:
        if not vectorstore:
            url = input("\nEnter the URL of the webpage to analyze (or 'quit' to exit): ").strip()
            
            if url.lower() == 'quit':
                print("Exiting application.")
                break
                
            documents = load_webpage(url)
            if documents:
                splits = split_documents(documents)
                vectorstore = create_vectorstore(splits, args.model)
                rag_chain = setup_rag_chain(vectorstore, args.model)
                print("\nReady for questions! (Type 'new' for a new webpage or 'quit' to exit)")
        else:
            question = input("\nEnter your question: ").strip()
            
            if question.lower() == 'quit':
                print("Exiting application.")
                break
            elif question.lower() == 'new':
                vectorstore = None
                continue
            
            try:
                answer = rag_chain.invoke(question)
                print("\nAnswer:", answer)
            except Exception as e:
                print(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()

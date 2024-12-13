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
    """
    Load and parse webpage content.

    Args:
        url (str): The URL of the webpage to load.

    Returns:
        List: A list of documents containing the parsed webpage content. 
              Returns an empty list if an error occurs.

    Raises:
        Exception: If there is an error loading the webpage.
    """
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
    """
    Split documents into smaller chunks.

    This function takes a list of documents and splits each document into smaller chunks
    using the RecursiveCharacterTextSplitter. The chunks are created based on the specified
    chunk size and overlap.

    Args:
        documents (List): A list of documents to be split.

    Returns:
        List: A list of document chunks.

    Example:
        documents = ["This is a long document...", "Another long document..."]
        chunks = split_documents(documents)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    print(f"\nSplit documents into {len(splits)} chunks")
    return splits

def create_vectorstore(splits: List, model: str) -> Chroma:
    """
    Create and populate a vector store using Chroma and Ollama embeddings.
    Args:
        splits (List): A list of document splits to be embedded and stored.
        model (str): The name of the model to be used for generating embeddings.
    Returns:
        Chroma: An instance of the Chroma vector store populated with the provided documents.
    """
    from chromadb.config import Settings
    import chromadb
    
    embeddings = OllamaEmbeddings(model=model)
    collection_name = f"webpage_collection_{model}"
    
    # Configure Chroma settings
    chroma_settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True
    )
    
    # Initialize the client
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=chroma_settings
    )
    
    # Delete collection if it exists
    try:
        client.delete_collection(collection_name)
    except:
        pass  # Collection might not exist
    
    # Create vectorstore with configured settings
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name=collection_name,
        client_settings=chroma_settings
    )
    print("\nCreated vector store with Ollama embeddings")
    return vectorstore

def setup_rag_chain(vectorstore: Chroma, model: str) -> RunnablePassthrough:
    """
    Set up the RAG (Retrieval-Augmented Generation) chain for querying.
    This function initializes a retriever from the provided vectorstore, pulls the default RAG prompt,
    initializes an Ollama LLM with the specified model, and creates a RAG chain for querying.
    Args:
        vectorstore (Chroma): The vector store to use for retrieval.
        model (str): The model name to use for the Ollama LLM. Examples include "mistral" or "gemma".
    Returns:
        RunnablePassthrough: The configured RAG chain ready for querying.
    """
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
    """Main application loop.
    This function runs the main application loop for a RAG (Retrieval-Augmented Generation) system
    that processes web page content and answers questions about it.
    The application flow:
    1. Takes a model name as command line argument (defaults to "llama3")
    2. Prompts user for a webpage URL
    3. Loads and processes the webpage content
    4. Accepts questions about the webpage content
    5. Provides AI-generated answers using the specified model
    6. Allows switching to a new webpage or quitting the application
    Command line arguments:
        --model: Ollama model to use (default: "llama3")
    User commands:
        'quit': Exits the application
        'new': Loads a new webpage
    Returns:
        None
    Raises:
        Various exceptions may be raised during webpage loading or answer generation
    """
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

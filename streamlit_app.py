#!/usr/bin/env python3
"""
Streamlit UI for the RAG application
"""

import streamlit as st
from rag_app import load_webpage, split_documents, create_vectorstore, setup_rag_chain

def main():
    st.title("Web Page Content Query System")
    st.write("Enter a URL to analyze and ask questions about its content")

    # Initialize session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'current_url' not in st.session_state:
        st.session_state.current_url = ""

    # URL input section
    url = st.text_input("Enter webpage URL:", key="url_input")
    
    if url and url != st.session_state.current_url:
        if st.button("Load Webpage"):
            with st.spinner("Loading webpage..."):
                documents = load_webpage(url)
                if documents:
                    splits = split_documents(documents)
                    st.session_state.vectorstore = create_vectorstore(splits)
                    st.session_state.rag_chain = setup_rag_chain(st.session_state.vectorstore)
                    st.session_state.current_url = url
                    st.success("Webpage loaded successfully!")
                else:
                    st.error("Failed to load webpage")

    # Question input section
    if st.session_state.vectorstore is not None:
        st.write("---")
        st.write("Ask questions about the webpage content:")
        question = st.text_input("Your question:", key="question_input")
        
        if question:
            if st.button("Get Answer"):
                try:
                    with st.spinner("Generating answer..."):
                        answer = st.session_state.rag_chain.invoke(question)
                        st.write("### Answer:")
                        st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

        if st.button("Clear Current Webpage"):
            st.session_state.vectorstore = None
            st.session_state.rag_chain = None
            st.session_state.current_url = ""
            st.experimental_rerun()

if __name__ == "__main__":
    main()

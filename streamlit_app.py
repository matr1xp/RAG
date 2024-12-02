#!/usr/bin/env python3
"""
Streamlit UI for the RAG application
"""

import streamlit as st
from rag_app import load_webpage, split_documents, create_vectorstore, setup_rag_chain

def main():
    # Set page config and custom CSS
    st.set_page_config(
        page_title="Web Page Content Query System",
        page_icon="🔍",
    )
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .stButton>button {
            color: white;
            background-color: #0e4c92;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
        }
        .clear-button>button {
            background-color: #d32f2f !important;
        }
        .success-message {
            color: #2e7d32;
            padding: 1rem;
            border-radius: 4px;
            background-color: #e8f5e9;
        }
        .error-message {
            color: #d32f2f;
            padding: 1rem;
            border-radius: 4px;
            background-color: #ffebee;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔍 Web Page Content Query System")
    st.markdown("""
        <div style='background-color: #e3f2fd; padding: 1rem; border-radius: 4px;'>
            <span style='color: #000000;'>Enter a URL to analyze and ask questions about its content</span>
        </div>
    """, unsafe_allow_html=True)

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
                    st.markdown("<div class='success-message'>Webpage loaded successfully! ✅</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='error-message'>Failed to load webpage ❌</div>", unsafe_allow_html=True)

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

        if st.button("Clear Current Webpage", key="clear-button", help="Reset the application state"):
            st.markdown("<style>.clear-button>button { background-color: #d32f2f !important; }</style>", unsafe_allow_html=True)
            st.session_state.vectorstore = None
            st.session_state.rag_chain = None
            st.session_state.current_url = ""
            st.experimental_rerun()

if __name__ == "__main__":
    main()

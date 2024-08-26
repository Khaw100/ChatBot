import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv() 
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

@st.cache_resource
def load_llm_and_embeddings(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": "Hello"},
            {"role": "model", "parts": "Great to meet you. What would you like to know?"},
        ]
    )
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return chat, embeddings

@st.cache_data
def load_faq_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    faq_pairs = content.split('Question:')
    data = []
    
    for pair in faq_pairs:
        if pair.strip():
            question_answer = pair.split('Answer:')
            if len(question_answer) == 2:
                question = question_answer[0].strip()
                answer = question_answer[1].strip()
                data.append({"question": question, "answer": answer})
    
    return pd.DataFrame(data)

@st.cache_resource
def create_vector_store_from_faq(faq_df, _embeddings):
    documents = [Document(page_content=f"Question: {row['question']}\nAnswer: {row['answer']}") for _, row in faq_df.iterrows()]
    vector_store = FAISS.from_documents(documents, _embeddings)
    return vector_store

def create_prompt_template():
    template = """You are a virtual assistant for PG&E, specifically designed to assist with Bill FAQs. 
    Please provide an accurate and concise answer to the following question based only on the context provided below. 
    If the context does not contain the relevant information, politely decline to answer and suggest visiting the Bill FAQs page for more details.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    return template

def main():
    st.title("PG&E Virtual Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    chat, embeddings = load_llm_and_embeddings()
    faq_df = load_faq_data('FAQ.txt')
    vector_store = create_vector_store_from_faq(faq_df, embeddings)
    retriever = vector_store.as_retriever()
    prompt_template = create_prompt_template()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context_docs = retriever.get_relevant_documents(prompt)
        context = " ".join([doc.page_content for doc in context_docs])

        final_prompt = prompt_template.format(context=context, question=prompt)
        response = chat.send_message(final_prompt).text

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

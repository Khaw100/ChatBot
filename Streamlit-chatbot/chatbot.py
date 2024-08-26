import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.schema import Document
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv() 
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

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

def create_vector_store_from_faq(faq_df, embeddings):
    documents = [Document(page_content=f"Question: {row['question']}\nAnswer: {row['answer']}") for _, row in faq_df.iterrows()]
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def create_prompt_template():
    template = """Please answer the following question based only on the provided context. 
    If the context does not contain the relevant information, politely decline to answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    return prompt_template

def custom_retrieval_chain(chat, question, retriever, prompt_template):
    context_docs = retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in context_docs])
    final_prompt = prompt_template.format(context=context, question=question)
    response = chat.send_message(final_prompt)
    return response.text

chat, embeddings = load_llm_and_embeddings()
faq_df = load_faq_data('FAQ.txt')
vector_store = create_vector_store_from_faq(faq_df, embeddings)
prompt_template = create_prompt_template()
retriever = vector_store.as_retriever()

st.title("Chatbot with Retrieval-based Context")

question = st.text_input("Ask your question:")

if st.button("Get Answer"):
    if question:
        response = custom_retrieval_chain(chat, question, retriever, prompt_template)
        st.write("**Answer:**", response)
    else:
        st.write("Please enter a question.")

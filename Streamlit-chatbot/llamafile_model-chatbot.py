import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.llms.llamafile import Llamafile
from langchain.prompts import PromptTemplate


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

faq_df = load_faq_data('FAQ.txt')

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents = [Document(page_content=f"Question: {row['question']}\nAnswer: {row['answer']}") for _, row in faq_df.iterrows()]
vector_store = FAISS.from_documents(documents, embeddings)

llm = Llamafile()

template = """
Based on the following context:

{context}


Please answer the question: {question}
"""
prompt_template = PromptTemplate(template=template, input_variables=["question", "context"])

def generate_response(question):
    retrieved_docs = vector_store.similarity_search(question, k=1)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    final_prompt = prompt_template.format(question=question, context=context)
    response = llm.invoke(final_prompt)
    return response

question = "Why is my gas bill so high?"
result = generate_response(question)
print(result)
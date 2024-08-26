import pandas as pd
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv() 
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

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

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                max_length=512,
                temperature=0.7)
llm = HuggingFacePipeline(pipeline=pipe)

template = """
Based on the following context:

{context}

Please answer the question: {question}
"""

prompt_template = PromptTemplate(
    template=template,
    input_variables=["question", "context"]
)

def truncate_text(text, max_length):
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    if tokens.shape[1] > max_length:
        tokens = tokens[:, :max_length]
        text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return text

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

def generate_response(question):
    retrieved_docs = vector_store.similarity_search(question, k=2)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    truncated_context = truncate_text(context, max_length=400)
    final_prompt = prompt_template.format(question=question, context=truncated_context)
    response = llm(final_prompt, max_new_tokens=150)
    return response[0]['generated_text'] if isinstance(response, list) else response

question = "Mengapa tagihan gas saya begitu tinggi?"
result = generate_response(question)

print(result)
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.schema import Document

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

# LLM
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, model_kwargs={"do_sample": True, "temperature": 0.8, "max_length": 512})
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt template
template = """
Hello! I’m PG&E Bill Assistant, here to help you with any questions related to your PG&E bill. 
You can ask me anything about billing, payments, or services provided by PG&E. How can I assist you today?

Here is the information I found based on your question:

{context}

Your Question: {question}

Answer:
"""

prompt_template = PromptTemplate(
    template=template,
    input_variables=["question", "context"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

question = "Why is my gas bill so high?"
result = qa_chain.run(question)

print(result)

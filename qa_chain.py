from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_qa_chain(llm, retriever):
    prompt_template = """
    You are MovieBot, an expert in movie trivia. Use the following context to answer the question:
    {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

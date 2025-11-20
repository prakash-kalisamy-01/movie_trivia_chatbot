import gradio as gr
from pdf_loader import load_and_split_pdf
from vector_store import create_vector_store
from llm_setup import get_llm
from qa_chain import build_qa_chain
from memory import get_chat_memory
from langsmith_setup import enable_langsmith

# Enable LangSmith
enable_langsmith()

# Load PDF and create vector store
chunks = load_and_split_pdf("movies_trivia.pdf")
vectorstore = create_vector_store(chunks)
retriever = vectorstore.as_retriever()

# LLM and QA chain
llm = get_llm()
qa_chain = build_qa_chain(llm, retriever)
chat_history = get_chat_memory()

# Gradio UI
def chat_fn(user_input):
    response = qa_chain.run(user_input)
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response)
    return response

def clear_history():
    chat_history.clear()
    return "History cleared."

with gr.Blocks() as demo:
    gr.Markdown("## Hello, I am MovieBot, your movie trivia expert. Ask me anything about films!")
    user_input = gr.Textbox(label="Your Question")
    output = gr.Textbox(label="MovieBot's Answer")
    submit_btn = gr.Button("Submit")
    clear_btn = gr.Button("Clear History")

    submit_btn.click(chat_fn, inputs=user_input, outputs=output)
    clear_btn.click(clear_history, outputs=output)

demo.launch()

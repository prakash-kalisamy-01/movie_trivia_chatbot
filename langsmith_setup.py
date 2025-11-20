import os
from dotenv import load_dotenv
from langsmith import Client

def enable_langsmith():
    load_dotenv()
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project_name = os.getenv("LANGCHAIN_PROJECT")
    client = Client(api_key=api_key)
    client.set_project(project_name)
    return client

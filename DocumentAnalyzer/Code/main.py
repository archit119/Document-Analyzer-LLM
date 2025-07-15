# Standard libraries
import os
import glob
from dotenv import load_dotenv

# Gradio
import gradio as gr

# LangChain loaders
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

# LangChain text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain schema
from langchain.schema import Document

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chat model
from langchain_openai import ChatOpenAI

# Vector store
from langchain_chroma import Chroma

# LangChain memory & chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Plotting & visualization
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go

from langchain.prompts import PromptTemplate


# price is a factor for our company, so we're going to use a low cost model
MODEL = "gpt-4o-mini"
db_name = "./chroma_db"

load_dotenv(override=True)
#os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

loader = PyPDFLoader(r"C:\Users\archi\OneDrive\Desktop\VS Code Scipts\DocumentAnalyzer\Files\CrossBorder_Payments.pdf")
documents = loader.load()

print("Loaded documents:", len(documents))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

#Prompt Template
prompt_template = """
You are a helpful assistant. Use ONLY the context below to answer the user's question.
If you don't know the answer from the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

QA_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# create a new Chat with OpenAI
llm = ChatOpenAI(
    temperature=0, 
    model_name=MODEL 
)

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever(search_kwags={"k": 15})

# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retriever, 
    memory=memory, 
    combine_docs_chain_kwargs={"prompt": QA_PROMPT} 
)

def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]

view = gr.ChatInterface(chat, type = "messages").launch(inbrowser=True)
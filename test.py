import os
import platform

import config # key 정보

import openai
import chromadb
import langchain
import tiktoken

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import PyPDFLoader


os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory="result", embedding_function=embeddings)

model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
chain = ChatVectorDBChain.from_llm(model, vectordb, return_source_documents=True)

history = []
while True:
    query = input().strip()
    result = chain({"question": query, "chat_history": history})
    history.append((query, result['answer']))
    print('Answer : ', result['answer'])
    # print(result)
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

loader = PyPDFLoader('data8.pdf')
data = loader.load_and_split()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

print(f'예상되는 토큰 수 {num_tokens_from_string(data[0].page_content, "cl100k_base")}')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(data)
# print(docs[3])

print(f"문서에 {len(docs)}개의 문서를 가지고 있습니다.")

persist_directory="result"

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
vectordb.persist()

# model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# chain = ChatVectorDBChain.from_llm(model, vectordb, return_source_documents=True)

# history = []
# while True:
#     query = input().strip()
#     result = chain({"question": query, "chat_history": history})
#     history.append((query, result['answer']))
#     print('Answer : ', result['answer'])
#     print(result)
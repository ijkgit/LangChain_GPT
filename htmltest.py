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

loader = UnstructuredURLLoader(urls=['https://www.google.com/search?q=%EC%82%AC%EB%82%B4+%EC%B1%97%EB%B4%87&oq=%EC%82%AC%EB%82%B4+%EC%B1%97%EB%B4%87&aqs=chrome..69i57j35i39l2j0i131i433i512j46i131i199i433i465i512l2j0i3l2j0i131i433i512j46i131i433i512.1412j0j7&sourceid=chrome&ie=UTF-8'])
data = loader.load_and_split()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

print(f'예상되는 토큰 수 {num_tokens_from_string(data[0].page_content, "cl100k_base")}')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(data)
print(docs)

# print(f"문서에 {len(docs)}개의 문서를 가지고 있습니다.")

# persist_directory="result"

# os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# vectordblist = []
# embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
# vectordb.persist()

# model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# chain = ChatVectorDBChain.from_llm(model, vectordb, return_source_documents=True)
# # print(chain)
# # query = "오늘 학식 뭐야?"
# # result = chain({"question": query, "chat_history": []})

# history = []
# while True:
#     query = input().strip()
#     result = chain({"question": query, "chat_history": history})
#     history.append((query, result['answer']))
#     print('Answer : ', result['answer'])
#     print(result)
#     # print('source', result['source_documents'])
    
# # text_splitter = RecursiveCharacterTextSplitter(chunk_size=)
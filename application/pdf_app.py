from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, pinecone, Weaviate, FAISS
from langchain.document_loaders import DirectoryLoader

from langchain.vectorstores import Chroma
from pymongo import MongoClient
from chromadb.utils  import embedding_functions
from flask import Flask, request, render_template, send_from_directory, jsonify
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import pdfplumber
import os
os.environ["OPENAI_API_KEY"] = "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"

MONGO_URI = "mongodb://192.168.50.84:27016/docsgpt"
PERSIST_DIR = "chroma_db"

mongo = MongoClient(MONGO_URI)
#mongo db name
db = mongo["docsgpt"]
#collection name vectors
vectors_collection = db["vectors"]

app = Flask(__name__)

def split_text():
    raw_text = ''
    with pdfplumber.open('D:/内部资料.pdf') as pdf:
        for page in pdf.pages:
            text =  page.extract_text()
            if text:
                raw_text += text

# print(raw_text)
def init_chroma():
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200)
    # texts = text_splitter.split_text(raw_text)   

    loader = DirectoryLoader('./Notion_DB/', glob='**/*.md')
    # 将数据转成 document 对象，每个文件会作为一个 document
    documents = loader.load()
    split_docs = text_splitter.split_documents(documents)
 
    # embedding
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="tuhailong/chinese-roberta-wwm-ext")
    # docsearch = FAISS.from_texts(texts, embeddings)
    global docsearch
    docsearch = Chroma.from_documents(documents=split_docs, embedding= embeddings, persist_directory= PERSIST_DIR )


def query():
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    # Chroma.add_documents()
    query = "这篇文件的主要内容?"
    docs = docsearch.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    print(result)
    
@app.route("/api/search", methods=["POST"])
def api_search():
    
    pass    
    

if __name__ == '__main__':
    pass


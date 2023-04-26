"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
# 序列化过程叫作 pickle
import pickle


# Here we load in the data in the format that Notion exports it in.
ps = list(Path("./Notion_DB/").glob("**/*.md"))

data = []
sources = []
for p in ps:
    with open(p, 'r', encoding='utf-8') as f:
        data.append(f.read())
    sources.append(p)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
# chunk_size=1000 表示一个 chunk 有 1000 个字符，而 chunk_overlap 表示下一个 chunk 会重复上一个 chunk 最后 200 字符的内容，
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

print(docs)
# Here we create a vector store from the documents and save it to disk.
# 文本 chunk 做向量化
store = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key="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"), metadatas=metadatas)
print(store.index)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
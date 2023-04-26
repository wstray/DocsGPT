# from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# embeddings = model.encode(sentences)
# print(embeddings)

import certifi
from urllib.request import urlopen


urlopen('https://example.com/bar/baz.html', cafile=certifi.where())
from sentence_transformers import SentenceTransformer, util
from flask import Flask, jsonify, request
from langchain.embeddings.huggingface import

from transformers import AutoTokenizer, AutoModel

def test():
    model = AutoModel.from_pretrained("tuhailong/chinese-roberta-wwm-ext")
    tokenizer = AutoTokenizer.from_pretrained("tuhailong/chinese-roberta-wwm-ext")
    sentences_str_list = ["今天天气不错的","天气不错的"]
    inputs = tokenizer(sentences_str_list,return_tensors="pt", padding='max_length', truncation=True, max_length=32)
    outputs = model(**inputs)
    print(outputs)

model_path = "tuhailong/chinese-roberta-wwm-ext"
def evalute_sentence():
    model = SentenceTransformer(model_path)
    s1 = "我喜欢你"
    s2 = "我不是不喜欢你"
    if s1 and s2:
        embedding1 = model.encode(s1, convert_to_tensor=True)
        print(embedding1)
        embedding2 = model.encode(s2, convert_to_tensor=True)
        print(embedding2)
        similarity = util.cos_sim(embedding1, embedding2).tolist()
        print(f'similarity={similarity}')
        # return jsonify({"code": 200, "msg": "预测成功", "data": similarity})
    else:
        # return jsonify({"code": 400, "msg": "缺少字段"})
        pass
    
    
if __name__ == '__main__':
    evalute_sentence()
    # test()

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import re
import os

def clean_text(text):
    if pd.isna(text): return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9.,!? ]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def load_and_prep(df_path):
    df = pd.read_csv(df_path, sep="\t")
    df["vendor_name"] = df["vendor_name"].fillna("без бренда")
    df["shop_category_name"] = df["shop_category_name"].fillna("")
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].apply(clean_text)
    df["full_text"] = df["title"] + " | " + df["shop_category_name"] + " | " + df["vendor_name"] + " | " + df["description"]
    return df

def main():
    df_test = load_and_prep("test.tsv") 
    
    db_embs = np.load("train_embeddings.npy")
    db_targets = pd.read_csv("train_targets.csv")
    
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    model.eval()
    
    test_texts = df_test["full_text"].values
    embs = []
    
    with torch.no_grad():
        for i in range(0, len(test_texts), 32):
            inputs = tokenizer(test_texts[i:i+32].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
            out = model(**inputs)
            e = out.last_hidden_state[:, 0, :]
            e = torch.nn.functional.normalize(e, p=2, dim=1)
            embs.append(e.cpu().numpy())
            
    test_embs = np.vstack(embs)
    
    sims = np.dot(test_embs, db_embs.T)
    best_idx = np.argmax(sims, axis=1)
    
    df_test["department_id"] = db_targets.iloc[best_idx]["department_id"].values
    df_test["category_id"] = db_targets.iloc[best_idx]["category_id"].values
    
    df_test[["department_id", "category_id"]].to_csv("prediction.csv", index=False)

if __name__ == "__main__":
    main()
# 簡單 RAG

這個專案是一個簡單版的 RAG 系統。RAG 是 Retrieval-Augmented Generation 的縮寫，意思是先從資料裡找出和問題有關的內容，再把找到的內容拿來輔助回答問題。

目前這個版本使用 `sentence-transformers` 產生語意向量，並搭配關鍵字命中分數做 hybrid retrieval。它沒有使用外部向量資料庫，而是把文章段落的 embedding 保存在記憶體中，適合用來理解 RAG 的基本流程。

## 目前流程

1. 將內建文章切成多個段落。
2. 使用 `paraphrase-multilingual-MiniLM-L12-v2` 將段落轉成 embedding。
3. 讓使用者輸入查詢文字。
4. 將查詢文字轉成 embedding，計算與各段落的 semantic similarity。
5. 同時抽取英文與中文關鍵字，計算 keyword match score。
6. 使用 hybrid score 排序結果：

```text
hybrid_score = 0.8 * semantic_score + 0.2 * keyword_score
```

7. 顯示分數最高的 3 段內容，並標出命中的關鍵字。

## 安裝

建議使用 Python virtual environment。

```bash
pip install -r requirements.txt
```

第一次執行時，程式會下載 `sentence-transformers` 模型，並快取到 `./hf_cache`。

## 執行

```bash
python rag.py
```

執行後輸入想查詢的關鍵字或問題，例如：

```text
RAG 是什麼
embedding
retrieval 找錯
```

直接按 Enter 送出空白查詢即可結束程式。

## 適合學習的重點

- chunking：如何把資料切成段落
- embedding：如何把文字轉成語意向量
- retrieval：如何找出和問題最相關的段落
- keyword matching：如何補強明確關鍵字命中
- hybrid ranking：如何結合語意相似度與關鍵字分數


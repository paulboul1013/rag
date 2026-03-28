import re
from sentence_transformers import SentenceTransformer, util

article = """
RAG 是 Retrieval-Augmented Generation 的縮寫，中文常翻成檢索增強生成。它的核心概念不是只依靠模型參數中的知識，而是先從外部資料中找出相關內容，再根據這些內容回答問題。

一個典型的 RAG system 通常分成幾個步驟。第一步是準備資料，像是 documents、筆記、FAQ 或網頁內容。第二步是把這些資料切成多個小片段，這個動作通常叫做 chunking。

為什麼要切塊？因為一整篇文章通常太長，直接丟給模型不但浪費 token，也不容易精準找到答案。切成小片段之後，system 比較容易找出和問題最相關的部分。

在比較進階的做法裡，每個片段都會被轉成向量，這個過程叫做 embedding。當使用者發問時，問題也會被轉成向量，再去資料庫中尋找最接近的片段，這一步叫做 retrieval。

找到片段之後，系統不會直接把整個資料庫交給模型，而是只把最相關的幾段內容放進 prompt。這樣模型回答時，就能根據外部資料作答，而不是只靠自己記憶。

RAG 有一個很大的優點，就是可以降低模型胡亂回答的機率，也就是所謂的 hallucination。當模型手上有明確的參考片段時，答案通常會更穩定，也更容易追蹤來源。

不過 RAG 也不是萬能的。如果 retrieval 階段找錯片段，後面的 generation 也會跟著出錯。所以很多時候，RAG 的品質不只取決於模型本身，也取決於 chunking 方式、檢索方法，以及資料是否乾淨。
"""

def normalize_text(text):
    text=text.lower()
    text=re.sub(r"\s+"," ",text)
    return text.strip()

def extract_english_words(text):
    text=normalize_text(text)
    return re.findall(r"[a-z0-9]+",text)

def extract_chinese_terms(text):
    text=normalize_text(text)
    return re.findall(r"[\u4e00-\u9fff]+", text)

paragraphs=[p.strip() for p in article.split("\n\n") if p.strip()]
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2",
                            cache_folder="./hf_cache",   # 可選
                            local_files_only=False       # 第一次允許下載
    )

corpus_embeddings = model.encode_document(
    paragraphs,
    convert_to_tensor=True,
    normalize_embeddings=True
)
# print("total paragraphs:",len(paragraphs))
# for i,p in enumerate(paragraphs,start=1):
#     print(f"\n--- paragraph {i}")
#     print(p)


def score_paragraph(paragraph,keywords):
    normalized_paragraph=normalize_text(paragraph)
    
    english_words=extract_english_words(paragraph)
    english_word_set=set(english_words)

    score=0
    matched_keywords=[]

    for kw in keywords:
        kw=normalize_text(kw)

        direct_count=normalized_paragraph.count(kw)
        if direct_count > 0:
            score+=direct_count*3
            matched_keywords.append(kw)
            continue

        if re.fullmatch(r"[a-z0-9]+",kw):
            if kw in english_word_set:
                score+=2
                matched_keywords.append(kw)

    matched_keywords = list(dict.fromkeys(matched_keywords))
    return score,matched_keywords

def parse_keywords(query):
    # return [word.strip().lower() for word in query.split() if word.strip()]

    query=normalize_text(query)

    english_terms=re.findall(r"[a-z0-9]+",query)
    chinese_terms=re.findall(r"[\u4e00-\u9fff]+",query)

    keywords=english_terms+chinese_terms
    return keywords


def semantic_search_paragraphs(query,paragraphs,corpus_embeddings,top_k=3):
    query_embedding=model.encode_query(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    hits=util.semantic_search(
        query_embedding,
        corpus_embeddings,
        top_k=top_k
    )[0]

    results=[]
    for hit in hits:
        idx=hit["corpus_id"]
        score=float(hit["score"])
        paragraph=paragraphs[idx]
        results.append((idx+1,score,paragraph))

    return results



top_k=3

def highlight_text(text,keywords):
    highlighted=text
    for kw in keywords:
        pattern=re.compile(re.escape(kw),re.IGNORECASE)
        highlighted = pattern.sub(
            lambda m: f"\033[30;43m{m.group(0)}\033[0m",
            highlighted
        )
        
    return highlighted


while True:
    query=input("input key word:").strip()
    if not query:
        break
    semantic_results=semantic_search_paragraphs(
        query,
        paragraphs,
        corpus_embeddings,
        top_k=top_k
    )
    keywords=parse_keywords(query)
    if not semantic_results:
        print("can't find related paragraphs")
    else:
        print("\nmost related paragraphs: ")
        for idx,score,paragraph in semantic_results:
            print(f"\n--- paragraph {idx} ---")
            print("semantic score: ",round(score,4))
            print(highlight_text(paragraph,keywords))
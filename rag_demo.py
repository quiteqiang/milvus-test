import os
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient

load_dotenv()

api_key = os.getenv("MOONSHOT_API_KEY")
if not api_key:
    raise ValueError("未找到 MOONSHOT_API_KEY，请检查 .env 文件")
print(f"API Key 已加载: {api_key[:10]}...{api_key[-4:]}")

# ─── 1. 初始化 KIMI 客户端 ────────────────────────
client_llm = OpenAI(
    api_key=api_key,
    base_url="https://api.kimi.com/coding/v1",
)

# ─── 2. 初始化本地 Milvus ──────────────────────────
client_db = MilvusClient("./milvus_rag.db")

COLLECTION_NAME = "rag_docs"

# 预先获取一个 embedding，自动确定向量维度
# 注意：当前端点对 embedding 模型名不敏感，实际使用默认模型
_test_embedding = client_llm.embeddings.create(
    model="kimi-embedding",
    input=["test"],
).data[0].embedding
EMBEDDING_DIM = len(_test_embedding)
print(f"Embedding 维度: {EMBEDDING_DIM}")

if COLLECTION_NAME not in client_db.list_collections():
    client_db.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBEDDING_DIM,
    )

# ─── 3. 文本向量化函数 ────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client_llm.embeddings.create(
        model="kimi-embedding",  # 当前端点对模型名不敏感，实际使用默认 Embedding 模型
        input=texts,
    )
    return [item.embedding for item in response.data]

# ─── 4. 加载并切分文档（示例：硬编码文本） ──────────
documents = [
    "Milvus 是一款开源的向量数据库，专为高性能相似度检索设计。",
    "KIMI 是 Moonshot AI 推出的大语言模型，支持长文本理解。",
    "RAG 检索增强生成通过外部知识库提升大模型回答的准确性。",
    "向量数据库可以存储高维向量，并支持近邻搜索（ANN）。",
]

# 插入知识库（仅当 Collection 为空时插入，避免重复）
if client_db.get_collection_stats(COLLECTION_NAME)["row_count"] == 0:
    vectors = embed_texts(documents)
    client_db.insert(
        collection_name=COLLECTION_NAME,
        data=[
            {"id": i, "vector": vectors[i], "text": documents[i]}
            for i in range(len(documents))
        ],
    )
    print(f"已插入 {len(documents)} 条文档")
else:
    print("文档已存在，跳过插入")

# 加载 Collection 到内存，否则无法搜索
client_db.load_collection(COLLECTION_NAME)
print("Collection 已加载到内存")

# ─── 5. 检索函数 ───────────────────────────────────
def retrieve(query: str, top_k: int = 2) -> list[str]:
    query_vector = embed_texts([query])[0]
    results = client_db.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
    )
    ids = [hit["id"] for hit in results[0]]
    # 根据 id 取回原文
    rows = client_db.get(
        collection_name=COLLECTION_NAME,
        ids=ids,
        output_fields=["text"],
    )
    return [row["text"] for row in rows]

# ─── 6. 生成回答 ───────────────────────────────────
def ask(query: str) -> str:
    contexts = retrieve(query)
    prompt = f"""请根据以下参考信息回答问题。

参考信息：
{chr(10).join(f"- {c}" for c in contexts)}

问题：{query}
"""
    response = client_llm.chat.completions.create(
        model="kimi-for-coding",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手，只能基于提供的参考信息回答。"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

# ─── 7. 测试 ───────────────────────────────────────
if __name__ == "__main__":
    query = "Milvus 是什么？"
    print("问题：", query)
    print("回答：", ask(query))
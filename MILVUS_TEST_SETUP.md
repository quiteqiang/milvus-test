# 本地 RAG 测试方案：Milvus + KIMI（Moonshot）API

> 目标：所有数据与向量库存储在本地，仅调用 KIMI/Moonshot API 完成 Embedding 与 LLM 生成。

---

## 一、架构说明

```
本地文档 / 文本
    │
    ▼
KIMI Embedding API ──────→ 文本向量
    │
    ▼
Milvus Lite（本地文件） ────→ 向量检索
    │
    ▼
KIMI Chat API ───────────→ 生成回答
```

- **向量存储**：Milvus Lite（本地 `milvus_rag.db`）
- **向量化**：KIMI Embedding API
- **大模型生成**：KIMI Chat API
- **接入方式**：OpenAI-compatible SDK（`openai` Python 包）

---

## 二、前置依赖

- Python >= 3.8，建议 3.9 - 3.11
- KIMI/Moonshot API Key（从 https://platform.moonshot.cn 获取）
- 网络可访问 `https://api.moonshot.cn`

---

## 三、环境搭建

### 1. 创建并激活虚拟环境

```bash
cd /Users/qiangwang/Desktop/Dev/milvus-test
python -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install "pymilvus[milvus-lite]" openai python-dotenv
```

### 3. 配置 API Key

创建 `.env` 文件：

```env
MOONSHOT_API_KEY=your_moonshot_api_key_here
```

> 注意：`.env` 文件不要提交到 Git，已默认加入 `.gitignore`。

---

## 四、RAG 测试脚本

创建 `rag_demo.py`：

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient

load_dotenv()

# ─── 1. 初始化 KIMI 客户端 ────────────────────────
client_llm = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
)

# ─── 2. 初始化本地 Milvus ──────────────────────────
client_db = MilvusClient("./milvus_rag.db")

COLLECTION_NAME = "rag_docs"
EMBEDDING_DIM = 1024  # KIMI embedding 模型输出维度，请根据实际模型确认

if COLLECTION_NAME not in client_db.list_collections():
    client_db.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBEDDING_DIM,
    )

# ─── 3. 文本向量化函数 ─────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client_llm.embeddings.create(
        model="moonshot-v1-embedding",  # 请根据官方文档确认模型名称
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

# 插入知识库
vectors = embed_texts(documents)
client_db.insert(
    collection_name=COLLECTION_NAME,
    data=[
        {"id": i, "vector": vectors[i], "text": documents[i]}
        for i in range(len(documents))
    ],
)

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
        model="moonshot-v1-8k",  # 可换为 moonshot-v1-32k / moonshot-v1-128k
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手，只能基于提供的参考信息回答。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

# ─── 7. 测试 ───────────────────────────────────────
if __name__ == "__main__":
    query = "Milvus 是什么？"
    print("问题：", query)
    print("回答：", ask(query))
```

### 运行

```bash
source venv/bin/activate
python rag_demo.py
```

---

## 五、从本地文件构建知识库

如果需要加载 `.txt`、`.pdf`、`.md` 等文件，扩展脚本如下：

```python
import glob

def load_txt_files(folder: str) -> list[str]:
    texts = []
    for path in glob.glob(os.path.join(folder, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts

# 简单切分
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
```

---

## 六、Milvus Docker 版（可选）

如果后续需要测试多 Collection、Partition、标量过滤等高级功能，可切换到本地 Docker Standalone：

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.5/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
```

代码中只需替换连接：

```python
client_db = MilvusClient(uri="http://localhost:19530")
```

---

## 七、推荐项目结构

```
milvus-test/
├── venv/                    # Python 虚拟环境
├── data/                    # 本地文档目录
│   └── sample.txt
├── .env                     # API Key（不提交 Git）
├── .gitignore
├── milvus_rag.db            # Milvus Lite 数据文件
├── rag_demo.py              # RAG 完整流程示例
├── chunk_utils.py           # 文档切分工具
├── requirements.txt
└── README.md                # 项目说明
```

---

## 八、关键参数说明

| 参数 | 说明 | 备注 |
|------|------|------|
| `MOONSHOT_API_KEY` | KIMI API 密钥 | 从 https://platform.moonshot.cn 获取 |
| `base_url` | `https://api.moonshot.cn/v1` | OpenAI-compatible 接口 |
| `moonshot-v1-embedding` | Embedding 模型 | 输出维度需与 `EMBEDDING_DIM` 一致 |
| `moonshot-v1-8k` | 对话模型 | 可按需选择 32k / 128k |
| `EMBEDDING_DIM` | 向量维度 | 常见 1024 / 2560，请按模型文档填写 |

---

## 九、注意事项

1. **API Key 安全**：`.env` 文件不要上传到 GitHub。
2. **Embedding 维度**：创建 Collection 前务必确认 KIMI embedding 模型的输出维度。
3. **Token 与费用**：KIMI API 按 token 计费，测试时注意控制文档大小。
4. **网络**：调用 API 需要能访问 `api.moonshot.cn`。
5. **数据隐私**：文档向量存入本地 Milvus，原始文本是否上传取决于你的处理逻辑；本方案原始文本保留在本地。

---

## 十、扩展方向

- 接入 LangChain / LlamaIndex 简化 RAG 流程
- 使用 Attu 可视化 Milvus 数据：https://github.com/zilliztech/attu
- 添加 Web UI（Gradio / Streamlit）进行交互式问答
- 接入更复杂的文档解析（PDF、Markdown、HTML）

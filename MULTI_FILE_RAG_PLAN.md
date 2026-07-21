# 多文件 RAG 方案

基于当前 `rag_demo.py` 扩展，支持把多个本地文件导入 Milvus 作为知识库。

---

## 一、目标

把当前硬编码的 `documents` 列表：

```python
documents = [
    "Milvus 是一款开源的向量数据库...",
    "KIMI 是 Moonshot AI 推出的大语言模型...",
]
```

替换为从 `data/` 目录自动读取、切分、导入多个文件。

---

## 二、整体流程

```
data/
├── file1.txt
├── file2.md
└── file3.pdf
        │
        ▼
读取文件内容
        │
        ▼
切分成 chunks（固定长度 / 按段落 / 按 token）
        │
        ▼
每个 chunk 生成 embedding
        │
        ▼
存入 Milvus（带 metadata：文件名、chunk 序号、原文）
        │
        ▼
检索时返回最相关的 chunks
        │
        ▼
作为参考信息传给 LLM 生成答案
```

---

## 三、推荐项目结构

```
milvus-test/
├── data/                        # 放原始文档
│   ├── milvus_intro.txt
│   ├── kimi_intro.txt
│   └── rag_intro.md
├── venv/
├── .env
├── .env.example
├── .gitignore
├── README.md
├── MILVUS_TEST_SETUP.md
├── MULTI_FILE_RAG_PLAN.md       # 本方案文档
├── rag_demo.py
├── loader.py                    # 新增：文件读取 + 切分
└── test_api_key.py
```

---

## 四、新增 loader.py

### 1. 读取不同格式文件

支持 `.txt`、`.md`，可选 `.pdf`（需安装 `PyPDF2` 或 `pymupdf`）：

```python
import os
import glob

DATA_DIR = "./data"

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        return read_txt(path)
    # if ext == ".pdf":
    #     return read_pdf(path)
    raise ValueError(f"不支持的文件类型: {ext}")

def load_all_files(folder: str = DATA_DIR) -> list[dict]:
    files = []
    for ext in ("*.txt", "*.md"):
        files.extend(glob.glob(os.path.join(folder, ext)))

    documents = []
    for path in sorted(files):
        content = read_file(path)
        documents.append({
            "filename": os.path.basename(path),
            "content": content,
        })
    return documents
```

### 2. 文本切分策略

#### 方案 A：固定长度切分（简单）

```python
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
```

#### 方案 B：按段落切分（保留语义）

```python
def chunk_by_paragraph(text: str, max_length: int = 500) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > max_length and current:
            chunks.append(current)
            current = p
        else:
            current += "\n" + p if current else p
    if current:
        chunks.append(current)
    return chunks
```

### 3. 组合成 chunks 列表

```python
def build_chunks(documents: list[dict], chunk_fn=chunk_text) -> list[dict]:
    chunks = []
    chunk_id = 0
    for doc in documents:
        texts = chunk_fn(doc["content"])
        for idx, text in enumerate(texts):
            chunks.append({
                "id": chunk_id,
                "filename": doc["filename"],
                "chunk_index": idx,
                "text": text,
            })
            chunk_id += 1
    return chunks
```

---

## 五、修改 rag_demo.py

### 1. 替换硬编码 documents

```python
from loader import load_all_files, build_chunks

# 加载并切分所有文件
raw_docs = load_all_files("./data")
chunks = build_chunks(raw_docs)

# 插入 Milvus（仅当 Collection 为空时）
if client_db.get_collection_stats(COLLECTION_NAME)["row_count"] == 0:
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    client_db.insert(
        collection_name=COLLECTION_NAME,
        data=[
            {
                "id": c["id"],
                "vector": vectors[i],
                "text": c["text"],
                "filename": c["filename"],
                "chunk_index": c["chunk_index"],
            }
            for i, c in enumerate(chunks)
        ],
    )
```

### 2. 更新 retrieve() 返回更多信息

```python
def retrieve(query: str, top_k: int = 3) -> list[dict]:
    query_vector = embed_texts([query])[0]
    results = client_db.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
    )
    ids = [hit["id"] for hit in results[0]]
    rows = client_db.get(
        collection_name=COLLECTION_NAME,
        ids=ids,
        output_fields=["text", "filename", "chunk_index"],
    )
    return [
        {
            "text": row["text"],
            "filename": row["filename"],
            "chunk_index": row["chunk_index"],
        }
        for row in rows
    ]
```

### 3. 更新 prompt 格式化

```python
def format_contexts(contexts: list[dict]) -> str:
    lines = []
    for ctx in contexts:
        lines.append(f"[来源: {ctx['filename']} - 第 {ctx['chunk_index']} 段]\n{ctx['text']}")
    return "\n\n".join(lines)

def ask(query: str) -> str:
    contexts = retrieve(query, top_k=3)
    prompt = f"""请根据以下参考信息回答问题。

参考信息：
{format_contexts(contexts)}

问题：{query}
"""
    # ... 调用 LLM
```

---

## 六、Milvus Collection Schema 调整

当前 `create_collection` 只传了 `dimension`，这会自动启用动态字段，可以存 `filename`、`chunk_index` 等字段。

如果想显式定义 schema，可以这样做：

```python
from pymilvus import MilvusClient, DataType

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
schema.add_field("text", DataType.VARCHAR, max_length=4096)
schema.add_field("filename", DataType.VARCHAR, max_length=256)
schema.add_field("chunk_index", DataType.INT64)

index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="FLAT",  # 小数据量用 FLAT 即可
    metric_type="COSINE",
)

client_db.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params,
)
```

> 注意：如果之前用的是 `create_collection(dimension=...)` 创建的 Collection，建议先删除 `milvus_rag.db` 再运行新代码。

---

## 七、依赖更新

如果加入 PDF 支持，安装额外依赖：

```bash
pip install pymupdf  # 或 PyPDF2
```

---

## 八、使用步骤

1. 创建 `data/` 目录
2. 把 `.txt`、`.md` 文件放进去
3. 创建 `loader.py` 实现读取和切分
4. 修改 `rag_demo.py` 导入 loader 并替换 documents
5. 删除旧的 `milvus_rag.db`（schema 有变化时）
6. 运行 `python rag_demo.py`

---

## 九、后续优化方向

- **更智能的切分**：按语义段落、按 token、按句子
- **PDF / Word / HTML 支持**：用 `pymupdf`、`python-docx`、`beautifulsoup4`
- **去重**：相同内容不重复插入
- **增量更新**：只导入新增或修改的文件
- **检索后重排序（Rerank）**：用更精确的模型对检索结果排序
- **引用来源**：在 LLM 回答中标注答案来自哪个文件的哪一段

---

## 十、风险与注意事项

1. **API 费用**：文件越多、chunk 越多，Embedding API 调用次数越多
2. **Milvus Lite 容量**：本地文件版适合中小数据量，大数据量建议 Docker Standalone
3. **维度一致性**：如果更换 Embedding 模型，必须删除旧数据库重建
4. **Python 版本**：继续注意 Python 3.12 可能的兼容性问题

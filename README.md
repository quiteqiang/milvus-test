# Milvus RAG Test Project

本地 RAG（检索增强生成）测试项目：使用 **Milvus Lite** 作为本地向量数据库，通过 **KIMI API** 完成文本嵌入与答案生成。

---

## Architecture

```
Local Documents / Text
        │
        ▼
KIMI Embedding API  ────→  Text Vectors
        │
        ▼
Milvus Lite (.db file)  ────→  Vector Search
        │
        ▼
KIMI Chat API  ────→  Generated Answer
```

- **Vector Database**: Milvus Lite (`./milvus_rag.db`)
- **Embedding API**: KIMI/Moonshot-compatible API
- **LLM API**: KIMI/Moonshot-compatible API
- **Client**: `openai` Python SDK with custom `base_url`

---

## Requirements

- Python >= 3.8 (recommended 3.9 - 3.11; 3.12 may work but is not officially supported by Milvus Lite)
- A valid `MOONSHOT_API_KEY` in `.env`
- Network access to `https://api.kimi.com/coding/v1`

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install "pymilvus[milvus-lite]" openai python-dotenv

# 3. Configure API key
cp .env.example .env
# Edit .env and set MOONSHOT_API_KEY=sk-...
```

---

## Run

### Test API key only

```bash
python test_api_key.py
```

This verifies that both chat and embedding APIs are reachable.

### Run the full RAG demo

```bash
python rag_demo.py
```

Expected output:

```
API Key 已加载: sk-kimi-kD...lUIm
Embedding 维度: 1024
已插入 4 条文档
Collection 已加载到内存
问题： Milvus 是什么？
回答： ...
```

---

## File Structure

```
milvus-test/
├── .env                 # API key (ignored by git)
├── .env.example         # API key template
├── .gitignore
├── README.md            # This file
├── MILVUS_TEST_SETUP.md # Detailed setup guide (Chinese)
├── rag_demo.py          # Full RAG pipeline
└── test_api_key.py      # API key sanity check
```

---

## Notes

- **Model names**: The current endpoint (`https://api.kimi.com/coding/v1`) appears to use a default embedding model regardless of the `model` parameter. The code uses `bge_m3_embed` for clarity, but any value may return embeddings from the same underlying model.
- **Embedding dimension**: The script auto-detects the dimension from a test embedding call and creates the Milvus collection accordingly.
- **Data persistence**: Vectors and original text are stored locally in `milvus_rag.db`. Delete this file to reset the knowledge base.
- **Security**: Never commit `.env` or any file containing a real API key. `.gitignore` already excludes them.

---

## References

- Milvus: https://milvus.io
- Milvus GitHub: https://github.com/milvus-io/milvus
- Moonshot/KIMI Platform: https://platform.moonshot.cn

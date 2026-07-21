import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("MOONSHOT_API_KEY")
if not api_key:
    raise ValueError("未找到 MOONSHOT_API_KEY，请检查 .env 文件")

print(f"API Key 已加载: {api_key[:10]}...{api_key[-4:]}")
print(f"Base URL: https://api.kimi.com/coding/v1/models")
print("-" * 40)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.kimi.com/coding/v1",
)

# ─── 测试 1：Chat Completions ────────────────────────
print("\n[测试 1] Chat Completions")
try:
    response = client.chat.completions.create(
        model="kimi-for-coding",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "你好，请用一句话介绍自己。"},
        ]
    )
    print("✅ Chat 调用成功")
    print(f"模型: {response.model}")
    print(f"回复: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Chat 调用失败: {e}")

# ─── 测试 2：Embeddings ───────────────────────────────
print("\n[测试 2] Embeddings")
try:
    response = client.embeddings.create(
        model="moonshot-v1-embedding",
        input=["这是一段测试文本"],
    )
    embedding = response.data[0].embedding
    print("✅ Embedding 调用成功")
    print(f"模型: {response.model}")
    print(f"向量维度: {len(embedding)}")
    print(f"前 5 个值: {embedding[:5]}")
except Exception as e:
    print(f"❌ Embedding 调用失败: {e}")

print("\n" + "-" * 40)
print("测试完成")

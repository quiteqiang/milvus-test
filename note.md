There are 4 benchmarks are measured while developing a Rag based system.

In Retriever side, Recall & precision, are the 2 benchmarks.
Context Recall     : wehther we get all the info/data needed for user's query
Context precisisoin: Out of top-5 answers, how many of them are relevent to the correct answer

In generation side with LLM.
1. Answer Relevency: whether the answer is relevent to user's query & how much of relevency it has;
2. Faithfullness   : Whether there are any make up sotries.


How to select the chunk size:
1. depends on the usage of Rag system. 
for conversation use, it ranges from 1024 to 2048.
for general document use, it has wider token rang.
for tech document, it will shrink to 1/4 of docuent.

Advanced Strategy:
1. Overlapping cut: neighbor chunks needs to have 10-20% of overlapping
2. Semantic chunking:  after embedding, group them, then do the chunking --> need to do some experiement


Enterprise Level Q&A system 

## Usage cae of Rag sytem
## Dealing with PDF tables. 
Backgroud: After scanning the PDf, there are some tabels are cut and seperated into 2 or multiple piecees. Using OCR to capture the information directly could cause some wrong info and misleading logic of the table.

### Solution:
1. Layout Analysis: using Layout-parse or PaddleOCR to detect the whole context and find the difference of table & context
2. Using good model to process the tables are way too complicated.
3. Pipeline: using **PyMuPDF** to extract charactters. for the un-recognized images to use OCR to catpure the info.
4. Using C

# Embedding模型如何选择?
## 选型原则是模型上限覆盖 chunk size 并留余量——窗口是容量约束，不是质量指标。

# Agentic RAG 和 RAG 的区别
```python
普通RAG（被动）：
用户问 → 检索1次 → 固定Top-K → LLM生成
（检索策略固定，不管结果好不好）

Agentic RAG（主动）：
用户问 → Agent分析 → 决定是否检索/检索什么/怎么检索
         → 评估结果 → 决定是否再检索
         → 循环直到信息充分 → 生成
```

# TODO
1. 完整的RAG系统的开发流程
2. 自己上手做一遍所有的流程
3. 搞清楚 Re-rank 是什么, 为什么需要Re-rank. Re-rank的机制是什么
4. 搞清楚什么是 Recall, Context Precesion, Relevency, Faithfullness
5. MCP server 怎么实现


# Multimodal Tourism RAG

一个可以从零构建并运行的多模态旅游问答 RAG 项目。项目使用 LangGraph 作为总调度器，按问题类型自动选择普通向量 RAG、GraphRAG 或 OCR/VLM 多模态 RAG。

## 你可以用它做什么

- 查询中国内地旅游基础信息，例如景点、路线、住宿、美食和注意事项。
- 使用 GraphRAG 做更综合的旅游分析，例如多城市比较、主题总结和景点关系分析。
- 对港澳台旅游 PDF 资料做 OCR/VLM 解析后，构建多模态检索问答。
- 通过一个命令行入口统一对话：`python System_Agent.py`。

## 项目结构

```text
multimodel_RAG/
├── System_Agent.py                  # 总入口：LangGraph 多 Agent 路由
├── config.py                        # 路径与环境变量加载
├── MultiRAG_environment.yml         # 主环境 YAML
├── vllm_environment.yml             # OCR/VLM 服务环境 YAML
├── .env.example                     # 环境变量模板
├── datasets/
│   ├── travel_guide.csv             # 内地旅游 CSV 原始数据
│   └── gang_ao_pdf/                 # 港澳台旅游 PDF 原始资料
├── Naive_RAG/
│   ├── create_vectorstore.py        # 从 CSV 构建 FAISS 向量库
│   └── rag_agent.py                 # 普通 RAG Agent
├── GraphRAG/
│   ├── create_graphrag_datasets.py  # 生成 GraphRAG 输入文本
│   ├── graphrag_agent.py            # GraphRAG Agent
│   └── tourist_graphrag/
│       ├── settings.yaml            # GraphRAG 配置
│       └── prompts/                 # GraphRAG 提示词
├── vlm/
│   ├── vlm_multimodel_rag.py        # PDF OCR/VLM 处理与索引构建
│   └── multi_model_agent.py         # 港澳台多模态 RAG Agent
├── pure_ocr/                        # OCR 相关实验脚本
└── eval/                            # 离线评测脚本和样例
```

## 从零开始运行

下面步骤默认你刚克隆项目，仓库中没有任何已生成的索引、OCR 结果或 GraphRAG 输出。

### 1. 创建 Conda 环境

优先使用项目自带 YAML：

```bash
conda env create -f MultiRAG_environment.yml
conda activate MultiRAG
```

如果你的系统无法完全复现 YAML，也可以使用轻量安装方式：

```bash
conda create -n MultiRAG python=3.11 -y
conda activate MultiRAG
pip install -r requirements-core.txt
```

### 2. 配置 API Key

复制环境变量模板：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，填入自己的密钥和模型地址：

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
QWEN_API_KEY=your-dashscope-api-key
GRAPHRAG_API_KEY=your-graphrag-chat-api-key
GRAPHRAG_API_BASE=https://your-chat-compatible-endpoint/v1
GRAPHRAG_MODEL=your-chat-model-name
VLLM_ENDPOINT=http://localhost:3000/v1/chat/completions
```

说明：

- `DEEPSEEK_API_KEY`：系统路由和普通对话模型使用。
- `QWEN_API_KEY`：DashScope Embedding 使用。
- `GRAPHRAG_API_KEY`、`GRAPHRAG_API_BASE`、`GRAPHRAG_MODEL`：GraphRAG 建图和查询使用。
- `VLLM_ENDPOINT`：仅在重新处理 PDF OCR/VLM 时需要。

## 构建索引

必须先构建索引，再运行完整系统。

### 1. 构建普通 FAISS 向量库

```bash
python Naive_RAG/create_vectorstore.py
```

生成结果：

```text
faiss_index/
```

### 2. 构建 GraphRAG 数据和索引

先把 CSV 转成 GraphRAG 输入文本：

```bash
python GraphRAG/create_graphrag_datasets.py
```

GraphRAG CLI 会读取 `GraphRAG/tourist_graphrag/.env` 中的变量。把项目根目录的 `.env` 复制过去：

```bash
cp .env GraphRAG/tourist_graphrag/.env
```

Windows PowerShell：

```powershell
Copy-Item .env GraphRAG/tourist_graphrag/.env
```

再执行 GraphRAG 索引：

```bash
graphrag index --root GraphRAG/tourist_graphrag
```

生成结果：

```text
datasets/travel_guide.txt
GraphRAG/tourist_graphrag/input/
GraphRAG/tourist_graphrag/output/
GraphRAG/tourist_graphrag/cache/
GraphRAG/tourist_graphrag/logs/
```

### 3. 构建港澳台多模态索引

如果只测试内地普通 RAG 或 GraphRAG，可以先跳过本步骤。若要回答港澳台 PDF 资料相关问题，需要先启动 vLLM OCR 服务，再调用项目脚本处理 PDF。

OCR/VLM 依赖较重，建议在 Linux + GPU 环境中单独创建服务环境：

```bash
conda env create -f vllm_environment.yml
conda activate vllm
```

如果环境中还没有 olmOCR 模型，先下载模型。下面示例把模型保存到项目根目录：

```bash
modelscope download \
  --model allenai/olmOCR-7B-0725-FP8 \
  --local_dir ./olmOCR-7B-0725-FP8
```

如果系统缺少 PDF 转图片和字体依赖，在 Ubuntu 上可安装：

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
```

#### 终端 1：启动 vLLM OCR 服务

保持这个终端不要关闭：

```bash
conda activate vllm
vllm serve ./olmOCR-7B-0725-FP8 \
  --served-model-name olmocr \
  --max-model-len 16000 \
  --port 3000
```

启动成功后，服务地址为：

```text
http://localhost:3000/v1/chat/completions
```

确保 `.env` 中有：

```env
VLLM_ENDPOINT=http://localhost:3000/v1/chat/completions
```

#### 终端 2：调用 vLLM 服务处理 PDF

```bash
conda activate MultiRAG
python vlm/vlm_multimodel_rag.py
```

常见生成结果：

```text
vlm/localworkspace/
vlm/pdf_path/
vlm/result_markdown/
```

## 启动问答系统

完成索引构建后运行：

```bash
python System_Agent.py
```

终端会提示输入问题。输入 `exit` 退出。

示例：

```text
北京有哪些适合第一次去的景点？
从上海出发有哪些旅游景点可以去？
香港有什么好玩的地方？
台北有哪些美食推荐？
```

## 可选：生成流程图

默认不会生成流程图。如果需要重新生成 LangGraph 流程图：

```bash
DRAW_AGENT_GRAPHS=1 python System_Agent.py
```

Windows PowerShell：

```powershell
$env:DRAW_AGENT_GRAPHS="1"
python System_Agent.py
```

生成的图片属于中间文件，已被 `.gitignore` 忽略。

## 离线评测

只评测路由：

```bash
python eval/run_eval.py --cases eval/cases/sample_cases.jsonl --mode router
```

调用样例中指定的专家 Agent：

```bash
python eval/run_eval.py --cases eval/cases/sample_cases.jsonl --mode expected_agent --judge
```

调用完整系统：

```bash
python eval/run_eval.py --cases eval/cases/sample_cases.jsonl --mode system --judge
```

评测输出会生成在：

```text
eval/outputs/
eval/reports/
eval/runs/
```

## 常见问题

### 缺少环境变量

确认 `.env` 已经从 `.env.example` 复制，并填写了真实 key。

### 找不到 FAISS 索引

先运行：

```bash
python Naive_RAG/create_vectorstore.py
```

### 找不到 GraphRAG parquet 文件

先运行：

```bash
python GraphRAG/create_graphrag_datasets.py
graphrag index --root GraphRAG/tourist_graphrag
```

### 港澳台问题无法检索

请确认已经启动 vLLM OCR 服务，并执行过：

```bash
python vlm/vlm_multimodel_rag.py
```

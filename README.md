# Multimodal Tourism RAG

A from-scratch multimodal tourism question-answering project built with LangGraph, FAISS, GraphRAG, and OCR/VLM-based PDF processing.

The system uses `System_Agent.py` as the main entry point. It routes each user query to the most suitable specialist agent:

- **Naive RAG** for basic mainland China tourism questions.
- **GraphRAG** for higher-level tourism analysis, comparison, and summarization.
- **Multimodal RAG** for Hong Kong, Macau, and Taiwan tourism questions based on PDF materials.

## Features

- LangGraph-based multi-agent routing.
- FAISS vector retrieval over tourism CSV data.
- GraphRAG indexing and query workflow for structured tourism analysis.
- OCR/VLM pipeline for converting tourism PDFs into Markdown.
- Command-line interactive QA interface.
- Offline evaluation scripts.

## Project Structure

```text
multimodel_RAG/
|-- System_Agent.py                  # Main entry: LangGraph multi-agent router
|-- config.py                        # Project paths and environment loading
|-- MultiRAG_environment.yml         # Main Conda environment
|-- vllm_environment.yml             # OCR/VLM service Conda environment
|-- .env.example                     # Environment variable template
|-- datasets/
|   |-- travel_guide.csv             # Raw mainland China tourism CSV data
|   `-- gang_ao_pdf/                 # Raw Hong Kong, Macau, and Taiwan PDF files
|-- Naive_RAG/
|   |-- create_vectorstore.py        # Build FAISS vector store from CSV
|   `-- rag_agent.py                 # Naive RAG agent
|-- GraphRAG/
|   |-- create_graphrag_datasets.py  # Generate GraphRAG input text
|   |-- graphrag_agent.py            # GraphRAG agent
|   `-- tourist_graphrag/
|       |-- settings.yaml            # GraphRAG configuration
|       `-- prompts/                 # GraphRAG prompts
|-- vlm/
|   |-- vlm_multimodel_rag.py        # PDF OCR/VLM processing and index building
|   `-- multi_model_agent.py         # Multimodal RAG agent
|-- pure_ocr/                        # OCR experiment scripts
`-- eval/                            # Offline evaluation scripts and sample cases
```

## Start From Scratch

The following steps assume a clean clone of the repository, with no generated indexes, OCR outputs, or GraphRAG runtime files.

### 1. Create the Main Conda Environment

Use the provided YAML file:

```bash
conda env create -f MultiRAG_environment.yml
conda activate MultiRAG
```

If the full YAML environment cannot be reproduced on your platform, use the lightweight fallback:

```bash
conda create -n MultiRAG python=3.11 -y
conda activate MultiRAG
pip install -r requirements-core.txt
```

### 2. Configure Environment Variables

Copy the template file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and fill in your own API keys and model endpoints:

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
QWEN_API_KEY=your-dashscope-api-key
GRAPHRAG_API_KEY=your-graphrag-chat-api-key
GRAPHRAG_API_BASE=https://your-chat-compatible-endpoint/v1
GRAPHRAG_MODEL=your-chat-model-name
VLLM_ENDPOINT=http://localhost:3000/v1/chat/completions
```

Variable usage:

- `DEEPSEEK_API_KEY`: used by the router and chat agents.
- `QWEN_API_KEY`: used by DashScope embeddings.
- `GRAPHRAG_API_KEY`, `GRAPHRAG_API_BASE`, `GRAPHRAG_MODEL`: used by GraphRAG indexing and querying.
- `VLLM_ENDPOINT`: used only when rebuilding the OCR/VLM PDF pipeline.

Do not commit `.env` or real API keys to GitHub.

## Build Indexes

You must build the required indexes before running the full system.

### 1. Build the FAISS Vector Store

```bash
python Naive_RAG/create_vectorstore.py
```

Generated output:

```text
faiss_index/
```

### 2. Build the GraphRAG Index

First generate the GraphRAG input text from the CSV dataset:

```bash
python GraphRAG/create_graphrag_datasets.py
```

The GraphRAG CLI reads environment variables from `GraphRAG/tourist_graphrag/.env`. Copy your project-level `.env` into that directory:

```bash
cp .env GraphRAG/tourist_graphrag/.env
```

On Windows PowerShell:

```powershell
Copy-Item .env GraphRAG/tourist_graphrag/.env
```

Then run GraphRAG indexing:

```bash
graphrag index --root GraphRAG/tourist_graphrag
```

Generated outputs:

```text
datasets/travel_guide.txt
GraphRAG/tourist_graphrag/input/
GraphRAG/tourist_graphrag/output/
GraphRAG/tourist_graphrag/cache/
GraphRAG/tourist_graphrag/logs/
```

### 3. Build the Hong Kong, Macau, and Taiwan Multimodal Index

You can skip this step if you only want to test mainland China Naive RAG or GraphRAG. To answer Hong Kong, Macau, and Taiwan PDF-based questions, start the vLLM OCR service first, then run the project script that processes PDFs.

OCR/VLM dependencies are heavy. A Linux machine with GPU support is recommended.

Create the OCR/VLM service environment:

```bash
conda env create -f vllm_environment.yml
conda activate vllm
```

If the olmOCR model is not available locally, download it:

```bash
modelscope download \
  --model allenai/olmOCR-7B-0725-FP8 \
  --local_dir ./olmOCR-7B-0725-FP8
```

If your system is missing PDF-to-image or font dependencies, install them on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
```

#### Terminal 1: Start the vLLM OCR Service

Keep this terminal running:

```bash
conda activate vllm
vllm serve ./olmOCR-7B-0725-FP8 \
  --served-model-name olmocr \
  --max-model-len 16000 \
  --port 3000
```

After startup, the service endpoint should be:

```text
http://localhost:3000/v1/chat/completions
```

Make sure `.env` contains:

```env
VLLM_ENDPOINT=http://localhost:3000/v1/chat/completions
```

#### Terminal 2: Process PDFs With the vLLM Service

```bash
conda activate MultiRAG
python vlm/vlm_multimodel_rag.py
```

Common generated outputs:

```text
vlm/localworkspace/
vlm/pdf_path/
vlm/result_markdown/
```

## Run the QA System

After building the required indexes, start the interactive QA system:

```bash
python System_Agent.py
```

Type your question in the terminal. Type `exit` to quit.

Example questions:

```text
What are good attractions for a first trip to Beijing?
What tourist destinations can I visit from Shanghai?
What are some fun places to visit in Hong Kong?
What food is recommended in Taipei?
```

## Optional: Generate Workflow Graphs

Workflow graphs are not generated by default. To regenerate LangGraph workflow images:

```bash
DRAW_AGENT_GRAPHS=1 python System_Agent.py
```

On Windows PowerShell:

```powershell
$env:DRAW_AGENT_GRAPHS="1"
python System_Agent.py
```

Generated graph images are intermediate files and are ignored by `.gitignore`.

## Offline Evaluation

Evaluate routing only:

```bash
python eval/run_eval.py --cases eval/cases/sample_cases.jsonl --mode router
```

Call the expected specialist agent from each test case:

```bash
python eval/run_eval.py --cases eval/cases/sample_cases.jsonl --mode expected_agent --judge
```

Call the full system workflow:

```bash
python eval/run_eval.py --cases eval/cases/sample_cases.jsonl --mode system --judge
```

Evaluation outputs are generated under:

```text
eval/outputs/
eval/reports/
eval/runs/
```

## Generated Files Ignored by Git

This repository is kept in a clean from-scratch state. The following generated files and directories should not be committed:

- `.env`
- `.venv/`
- `.idea/`
- `faiss_index/`
- `datasets/travel_guide.txt`
- `datasets/gang_ao_pdf/pdf_path/`
- `GraphRAG/tourist_graphrag/.env`
- `GraphRAG/tourist_graphrag/input/`
- `GraphRAG/tourist_graphrag/output/`
- `GraphRAG/tourist_graphrag/cache/`
- `GraphRAG/tourist_graphrag/logs/`
- `olmOCR-*/`
- `vlm/localworkspace/`
- `vlm/pdf_path*/`
- `vlm/result_markdown/`
- `pure_ocr/image_path/`
- `pure_ocr/output/`
- `pure_ocr/pdf_path/`
- `pure_ocr/result_markdown/`
- `eval/outputs/`
- `eval/reports/`
- `eval/runs/`
- `__pycache__/`
- `.ipynb_checkpoints/`

## Troubleshooting

### Missing Environment Variables

Make sure `.env` has been copied from `.env.example` and filled with real keys.

### FAISS Index Not Found

Run:

```bash
python Naive_RAG/create_vectorstore.py
```

### GraphRAG Parquet Files Not Found

Run:

```bash
python GraphRAG/create_graphrag_datasets.py
graphrag index --root GraphRAG/tourist_graphrag
```

### Hong Kong, Macau, or Taiwan Retrieval Does Not Work

Make sure the vLLM OCR service is running, then run:

```bash
python vlm/vlm_multimodel_rag.py
```

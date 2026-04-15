# CITL Factbook Assistant v3.0

> **Lead Developer:** Abdo Mohammed
> **Project Lead:** Doc McDowell — CITL, Renton Technical College

AI-powered course factbook query system built for offline classroom use. Uses local Ollama LLMs with Retrieval-Augmented Generation (RAG) to let instructors and students search institutional curriculum documents, syllabi, and course catalogs — no internet required.

## What It Does

- **RAG indexing** — automatically chunks and indexes PDF/DOCX course materials into a local vector store
- **Offline LLM query** — sends retrieved context + question to a local Ollama model (Llama 3, Mistral, etc.)
- **Multi-document search** — queries across the entire factbook corpus simultaneously
- **Tkinter GUI** — accessible to non-technical staff, works on any Windows 10/11 machine
- **USB-portable** — runs from a flash drive with no installation on the host machine

## Skills Trained (IT Workforce Portfolio)

| Skill | Job Board Keyword |
|-------|------------------|
| Retrieval-Augmented Generation | RAG, LLM, vector search |
| Local LLM deployment | Ollama, LLaMA, Mistral |
| Python/Tkinter desktop apps | Python, GUI development |
| Document parsing & NLP | PDF extraction, NLP pipeline |
| Offline AI systems | Edge AI, air-gapped deployment |

## Portfolio Output

Working RAG demo, custom-indexed corpus, and a technical writeup of the AI pipeline — all shareable as GitHub projects.

## Quick Start

```bash
pip install -r requirements.txt
python factbook_assistant_gui.py
```

Requires [Ollama](https://ollama.ai) running locally with at least one model pulled.


## Authors & Contributors

| Name | Role |
|------|------|
| **Doc McDowell** | Project Lead, CITL Director of Instructional Technology |
| **Abdo Mohammed** | Lead Developer — Factbook AI Engine & RAG Systems |
| **Wahaj Al Obid** | Lead Developer — Academic Advisor v2.0 |
| **Jerome Anti Porta** | Developer — UI/UX, App Integration |
| **Jonathan Reed** | Developer — LLMOps & Model Management |
| **Peter Anderson** | Developer — AV/IT Operations & Network Tools |
| **Will Cram** | Developer — Sync Systems & Portable Deployment |
| **William Grainger** | Developer — Technical Writing & Documentation Tools |
| **Mason Jones** | Developer — Staff Toolkit & Field Apps |

> Renton Technical College — Center for Instructional Technology & Learning (CITL)
> Department of IT & Cybersecurity Workforce Development

# 📰 AI Newsletter Agent

## 🚀 Live Demo

**Hugging Face Space:** https://huggingface.co/spaces/Siggmoid/NewsLetter

---

## Overview

AI Newsletter Agent is an autonomous multi-agent application that researches the latest AI news, analyzes and summarizes the most relevant articles, generates a professional HTML newsletter, and simulates email delivery in a single function call.

The system is built using **LangGraph** to orchestrate an end-to-end autonomous workflow with optional human oversight.

A single call to:

```python
run_newsletter_agent(goal)
```

executes the complete pipeline from planning to final newsletter generation.

---

## Features

* 🔍 Researches the latest AI news from the web
* 📝 Summarizes the top 5–7 relevant articles
* 🎨 Generates a clean HTML newsletter
* 📧 Simulates email delivery by generating the email subject and HTML content
* 🤖 Fully autonomous execution with one function call
* 👨‍💻 Human-in-the-Loop mode for manual approval
* 🔄 Self-reflection and critique to improve output quality
* 🌐 Interactive web interface
* ⚡ FastAPI backend with Docker deployment

---

## Agent Workflow

```
Goal
   │
   ▼
Planning
   │
   ▼
Web Research
   │
   ▼
Article Selection
   │
   ▼
Summarization
   │
   ▼
Newsletter Generation
   │
   ▼
Self Review & Critique
   │
   ▼
Revision (if needed)
   │
   ▼
Final HTML Newsletter
```

---

## Autonomous Reasoning Pipeline

The agent follows a structured reasoning process:

1. Understand the user's goal
2. Plan the execution strategy
3. Search for recent AI news
4. Select the most relevant articles
5. Generate concise summaries
6. Create a professional HTML newsletter
7. Critique and review its own output
8. Improve the newsletter if necessary
9. Produce the final newsletter

---

## Tools Used

The agent demonstrates tool usage throughout the workflow.

### Web Search

Retrieves the latest AI-related news articles.

### Summarization

Extracts key insights from each selected article.

### HTML Generator

Creates a clean and responsive newsletter.

### Self Critique

Evaluates completeness, readability, formatting, and article relevance before producing the final output.

---

## Human-in-the-Loop

The application supports two execution modes.

### Fully Autonomous

The agent completes the entire workflow without user intervention.

```
Planning
→ Research
→ Summarization
→ Review
→ HTML Generation
→ Output
```

### Human-in-the-Loop

Allows the user to review intermediate results before generating the final newsletter.

Examples include:

* Reviewing selected articles
* Editing summaries
* Approving the final newsletter

---

## Tech Stack

### Framework

* LangGraph

### Backend

* FastAPI

### LLM

* Google Gemini

### Search

* DuckDuckGO Search No api needed

### Frontend

* HTML
* CSS
* JavaScript

### Deployment

* Docker
* Hugging Face Spaces

---

## Project Structure

```
.
├── agent.py
├── app.py
├── config.py
├── tools.py
├── index.html
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
cd NewsLetter
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```
GOOGLE_API_KEY=your_key
```

Run the application

```bash
python app.py
```

or

```bash
uvicorn app:app --reload
```

---

## Docker

Build

```bash
docker build -t newsletter-agent .
```

Run

```bash
docker run -p 7860:7860 newsletter-agent
```

---

## Example Goal

```
Create a weekly newsletter on the latest AI agent news and send it to our subscribers.
```

The agent automatically:

* researches current AI news
* selects the most relevant articles
* generates summaries
* creates a polished HTML newsletter
* performs self-review
* outputs the final newsletter

---

## Future Improvements

* Email integration (SMTP / SendGrid)
* Newsletter scheduling
* Multiple newsletter templates
* User authentication
* PDF export
* RSS feed support
* Vector memory for personalized newsletters
* Multi-language newsletter generation

---

## License

This project is intended for educational and demonstration purposes.

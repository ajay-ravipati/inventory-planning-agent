---
title: InventoryPlanningAgent
emoji: 🚀
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
license: apache-2.0
---
# Agentic Warehouse Inventory Planner

Welcome to the **Agentic Inventory System**! This application demonstrates a production grade AI agent that can query structured SQL databases, retrieve unstructured policy documents using RAG, and maintain conversational memory to assist with supply chain and warehouse management.

It is built using **LangGraph**, **Gemini**, **SQLite**, and **ChromaDB**, with a **Streamlit** frontend and a **Docker** SDK.

## Try It Out (Example Questions)
To see the agent's capabilities, try copying and pasting some of these queries into the chat. 

**1. Forecasting & Reordering (SQL Analytics Tool)**
The agent calculates projected stockouts based on current stock, daily demand, and safety stock thresholds.
* *"How many red sarees do we need to order for next week?"*
* *"Which items are currently below safety stock?"*
* *"What shirts do we need to order for the next 14 days?"*

**2. Ad-hoc SQL Queries (Text-to-SQL Tool)**
The agent can translate natural language into read-only SQLite queries to summarize data.
* *"How many total products do we have from SUPP_B?"*
* *"What is the average unit cost of all our black pants?"*
* *"Which product has the highest daily demand?"*

**3. Supplier Policies (RAG / Vector Search Tool)**
The agent searches unstructured markdown documents to answer policy and contract questions.
* *"What is the lead time for SUPP_A?"*
* *"Does SUPP_B have a minimum order quantity?"*
* *"What is the return policy for defective items from SUPP_C?"*

**4. Multi-Step Reasoning (Combined Tools)**
The agent can use conversational memory to drill down into specifics.
* **User:** *"Which items are currently below safety stock?"*
* **Agent:** *(Lists the items)*
* **User:** *"What is the lead time for the supplier of the first item on that list?"*

**5. Scope Guardrail (Security)**
The agent features a dedicated LLM routing node to block off-topic requests.
* *"Can you write a Python script for me?"* -> *(The agent will politely decline and keep you focused on inventory).*

---

## Architecture Under the Hood

This is not a simple LLM wrapper; it is a stateful, multi-agent workflow:
* **LLM Engine:** Google Gemini.
* **Orchestration:** LangGraph state machine with dual-memory (separating internal tool-calling logs from the clean UI).
* **Data Sources:** 
  * Structured: SQLite `inventory.db` containing SKUs, categories, and demand metrics.
  * Unstructured: ChromaDB vector store containing supplier markdown documents.
* **Guard Node Classification:** A lightweight preliminary LLM call protects the main execution path from prompt injections and out-of-scope requests.
* **Error Handling:** Implements finite state machine loops to retry failed SQL queries up to 2 times before informing the user.

## RAGAS Evaluation
This agent's pipeline was rigorously tested using **RAGAS** framework, which yielded a final production score of 0.762
* Evaluated metrics include **Faithfulness** (hallucination checks), **Answer Relevancy**, **Context Precision**, and **Context Recall**.
* The evaluation script ensures that when the agent queries ChromaDB for supplier policies, it extracts the correct chunks and synthesizes them without fabricating numbers.


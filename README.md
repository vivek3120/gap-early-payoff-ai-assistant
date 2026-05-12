# GAP Early Payoff Intelligence Assistant

Streamlit app link:- https://gap-early-payoff-ai-assistant-qdyijeqqrmhz72pd3wuhxq.streamlit.app/

## Project Overview

This project is an end-to-end **GAP Early Payoff Intelligence solution** built using a fully free / open-source-friendly stack. The project combines **Microsoft Fabric**, **Power BI**, **Python**, and a lightweight **RAG-style AI assistant** to analyze GAP early payoff activity and help business users ask natural-language questions about risk, dealerships, review cases, and dashboard metrics.

The project was designed around a realistic finance/lending use case: identifying and explaining early payoff patterns for vehicle loans with GAP protection.

---

## Business Problem

Financial teams often need to monitor loans that are paid off earlier than expected, especially when GAP products are involved. Early payoff activity can indicate cases that require further review, such as:

- Loans paid off within a short period after origination
- High GAP amount exposure
- Missing dealer GAP company information
- Repeated early payoff activity from specific dealerships
- Cases that should be reviewed manually by finance or operations teams

This project helps business users understand these patterns through both a **Power BI dashboard** and an **AI assistant**.

---

## What I Built

The project has three major parts:

1. **Data Engineering Layer in Microsoft Fabric**
2. **Power BI GAP Early Payoff Report**
3. **AI/RAG Assistant deployed as a Streamlit app**

---

## Tools and Technologies Used

### Data Engineering and Reporting

- Microsoft Fabric
- Fabric Lakehouse
- Fabric Notebook
- Bronze / Silver / Gold architecture
- Power BI

### AI / RAG Assistant

- Python
- Streamlit
- pandas
- scikit-learn
- TF-IDF vectorization
- Cosine similarity retrieval
- Lightweight RAG-style question answering
- Model evaluation framework

### Deployment

- GitHub
- Streamlit Community Cloud

---

## Architecture

```text
Synthetic GAP Early Payoff Data
        ↓
Microsoft Fabric Lakehouse
        ↓
Bronze Layer
        ↓
Silver Layer
        ↓
Gold Layer
        ↓
Power BI Report
        ↓
Gold AI Context CSV
        ↓
Python RAG-style Assistant
        ↓
Streamlit Web App
        ↓
Model Evaluation Results

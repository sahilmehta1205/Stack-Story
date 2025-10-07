# 📚 Chat with your PDF (LangChain + Chroma + Streamlit)

This app lets you upload a PDF and chat with it using LangChain + Chroma and a lightweight open LLM (FLAN-T5).

## 🚀 Features
- PDF upload & text extraction
- Text chunking & vector search via Chroma
- Lightweight model (`google/flan-t5-base`) via Hugging Face Hub
- Fast & free to deploy on Streamlit Cloud

## 🛠️ Setup
1. Create a free Hugging Face account → https://huggingface.co
2. Get your access token → https://huggingface.co/settings/tokens
3. In Streamlit Cloud:
   - Go to **Secrets → New Secret**
   - Add key: `HUGGINGFACEHUB_API_TOKEN`
   - Paste your token as the value

## 🖥️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

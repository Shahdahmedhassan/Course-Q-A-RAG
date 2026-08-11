# Standard Course Q&A RAG System

## Contents
- `data/` — sample material for 3 courses (Python, Data Science, Machine Learning), each with a `.pdf`, `.csv`, `.docx`, and `.txt` file.
- `RAG_Course_QA_Pipeline.ipynb` — the notebook (runs on Kaggle/Colab/local Jupyter): Extract → Clean → Chunk → Embeddings → FAISS Vector Store → Retriever → LLM → **save the model**.
- `app.py` — a Streamlit app that loads the saved model and answers student questions with source attribution.
- `requirements.txt` — required packages.

## Running on Kaggle
1. Create a new Kaggle Notebook.
2. Upload the `data/` folder as a Kaggle Dataset (or place it directly under `/kaggle/working/`).
3. Upload `RAG_Course_QA_Pipeline.ipynb` or copy its cells in.
4. Update `DATA_DIR` in the relevant cell to match your data path.
5. Run all cells in order — at the end you'll get a `rag_model/` folder (and `rag_model.zip`).
6. Download `rag_model.zip` from Kaggle's Output tab.

## Running Streamlit locally
```bash
pip install -r requirements.txt
# unzip rag_model.zip next to app.py if you haven't already
streamlit run app.py
```

## Notes
- The included data is sample lecture content — replace it with your real course materials using the
  same structure: `data/<course_name>/file.pdf|.csv|.docx|.txt`
- Embedding model: `all-MiniLM-L6-v2` (free, lightweight).
- Default generation model: `google/flan-t5-base` (free, local). An optional cell shows how to swap in the OpenAI API for higher-quality answers.

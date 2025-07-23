# NLP Insights

A lightweight end-to-end NLP pipeline for product reviews:

1. **Data loading**  
2. **Preprocessing** (lemmatization, stop-word removal via spaCy)  
3. **Sentiment analysis** (VADER)  
4. **Keyphrase extraction** (RAKE & spaCy)  
5. **Embeddings + dimensionality reduction** (Sentence-Transformers + UMAP)  
6. **Clustering** (HDBSCAN)  
7. **Visualization** (Matplotlib & Plotly)

---

## 📁 Repository structure

nlp_insights/
│
├─ data/
│ └─ raw/
│ └─ Reviews.csv # raw Amazon review data
│
├─ notebooks/ # exploratory notebooks
│ └─ 01_load_data.ipynb
│
├─ pic/ # final notebook screenshots
│ ├─ umap_matplotlib.png
│ └─ vader_distribution.png
│
├─ src/ # source modules
│ ├─ data_loader.py
│ ├─ preprocessing.py
│ ├─ sentiment.py
│ └─ clustering.py
│
├─ main.py # orchestrates each step via CLI flags
├─ requirements.txt
└─ README.md


## 🚀 Installation

1. **Clone** the repo:

   ```bash
   git clone https://github.com/NikitaMarshchonok/nlp_insights.git
   cd nlp_insights
   ```


2. **Create & activate** a virtual environment:
  ```bash
    python3 -m venv .venv
    source .venv/bin/activate
```
3. **Install** dependencies:
```bash
    pip install -r requirements.txt
```
4.**Download** spaCy English model:
```bash
    python -m spacy download en_core_web_sm
```

📊 Sample Outputs
UMAP + HDBSCAN clustering (Matplotlib)
VADER Sentiment Distribution (Plotly)


📄 License
MIT © 2025 Nikita Marshchonok
Feel free to use, modify, and contribute!



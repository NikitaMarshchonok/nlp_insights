# src/clustering.py

from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.decomposition import PCA
import numpy as np

def load_model(model_name: str = 'all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

def encode_texts(texts, model=None):
    if model is None:
        model = load_model()
    return model.encode(texts, show_progress_bar=True)

def reduce_dimensionality(embeddings, n_components: int = 2, random_state: int = 42):
    """
    Try UMAP first; if it fails, fall back to PCA.
    """
    try:
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=min(15, len(embeddings)-1)
        )
        return reducer.fit_transform(embeddings)
    except Exception as e:
        print(f"UMAP failed ({e}), falling back to PCA")
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(embeddings)

def cluster_embeddings(embeddings_2d, min_cluster_size: int = 15):
    """
    Cluster embeddings with HDBSCAN, ensuring min_cluster_size >= 2 and <= n_samples.
    """
    n_samples = len(embeddings_2d)
    if n_samples < 2:
        # для одного элемента возвращаем метку 0
        return np.zeros(n_samples, dtype=int)
    # гарантируем корректный размер кластера
    mcs = max(2, min_cluster_size)
    mcs = min(mcs, n_samples)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs)
    return clusterer.fit_predict(embeddings_2d)

def pipeline_cluster(texts, model=None, umap_components: int = 2, min_cluster_size: int = 15):
    """
    Full pipeline: encode → reduce_dim → cluster
    Returns: 2D embeddings, labels
    """
    if model is None:
        model = load_model()
    embeddings = encode_texts(texts, model)
    embeddings_2d = reduce_dimensionality(embeddings, n_components=umap_components)
    labels = cluster_embeddings(embeddings_2d, min_cluster_size=min_cluster_size)
    return embeddings_2d, labels

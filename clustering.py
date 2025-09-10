import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from embedding import EmbeddingWrapper

class Clustering:
    def __init__(self, embedding_function: EmbeddingWrapper):
        self.embedding_function = embedding_function
    
    def hdbscan_clustering(self, documents: list[dict], min_cluster_size: int = 5, min_samples: int = 5) -> list[list[dict]]:
        if len(documents) == 0:
            return [[]]

        # embeddings = collection.get(ids=[doc["id"] for doc in documents], include=["embeddings"])["embeddings"]
        embeddings = self.embedding_function([doc["document"] for doc in documents])
        # convert to matrix, cast to double_t
        embeddings = np.array(embeddings)
        embeddings = embeddings.astype(np.double)

        # compute cosine distance matrix
        distance_matrix = cosine_distances(embeddings)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed')
        labels = clusterer.fit_predict(distance_matrix)

        clustered_documents = [[] for _ in range(labels.max() + 2)]
        for doc, label in zip(documents, labels):
            doc["metadata"]["hdbscan_label"] = int(label)
            clustered_documents[label].append(doc)
        
        return clustered_documents

    def kmeans_clustering(self, documents: list[dict], n_clusters: int = 5) -> list[list[dict]]:
        from sklearn.cluster import KMeans
        if len(documents) == 0:
            return [[]]

        embeddings = self.embedding_function([doc["document"] for doc in documents])

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(embeddings)

        clustered_documents = [[] for _ in range(n_clusters)]
        for doc, label in zip(documents, labels):
            doc["metadata"]["kmeans_label"] = int(label)
            clustered_documents[label].append(doc)
        
        return clustered_documents

    def smooth_sample_clusters(self, clusters: list[list[dict]], laplace_smoothing: float, top_k: int) -> list[dict]:
        sizes = [len(c) for c in clusters]
        sizes_dist = [n/sum(sizes) for n in sizes]
        smoothed_sizes_dist = [(s + laplace_smoothing) / (1 + len(sizes)*laplace_smoothing) for s in sizes_dist]
        smoothed_sizes = [int(max(1, s*top_k)) for s in smoothed_sizes_dist]
        cutoff_clusters = [c[:s+1] for c, s in zip(clusters, smoothed_sizes)]
        return [doc for c in cutoff_clusters for doc in c][:top_k]




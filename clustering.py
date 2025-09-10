import hdbscan
from sklearn.metrics.pairwise import cosine_distances

def hdbscan_clustering(collection, documents: list[dict], min_cluster_size: int = 5, min_samples: int = 5) -> list[list[dict]]:
    if len(documents) == 0:
        return [[]]

    embeddings = collection.get(ids=[doc["id"] for doc in documents], include=["embeddings"])["embeddings"]

    # compute cosine distance matrix
    distance_matrix = cosine_distances(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed')
    labels = clusterer.fit_predict(distance_matrix)

    clustered_documents = [[] for _ in range(labels.max() + 2)]
    for doc, label in zip(documents, labels):
        doc["metadata"]["hdbscan_label"] = int(label)
        clustered_documents[label].append(doc)
    
    return clustered_documents

def kmeans_clustering(collection, documents: list[dict], n_clusters: int = 5) -> list[list[dict]]:
    from sklearn.cluster import KMeans
    if len(documents) == 0:
        return [[]]

    embeddings = collection.get(ids=[doc["id"] for doc in documents], include=["embeddings"])["embeddings"]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embeddings)

    clustered_documents = [[] for _ in range(n_clusters)]
    for doc, label in zip(documents, labels):
        doc["metadata"]["kmeans_label"] = int(label)
        clustered_documents[label].append(doc)
    
    return clustered_documents

from typing import List
import polars as pl
from collections import defaultdict
import os
from tqdm import tqdm

import chromadb
from chromadb.utils import embedding_functions

from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever

from config import CHROMA_PERSIST, DATA_FILEPATH, EMBEDDING_MODEL


def get_client_collection(persist, name):
    if not os.path.exists(persist):
        os.makedirs(persist, exist_ok=True)
    client = chromadb.PersistentClient(path=persist)
    return client, client.get_or_create_collection(name=name)

def prepare_paragraphs(data_filepath: str):
    # "par_id","doc_id","document_url","document_date","document_netloc","document_title","paragraph_text","paragraph_order","paragraph_score","document_score"
    df = pl.read_csv(data_filepath)
    # first, sort by doc_id and paragraph_order
    df = df.sort(["doc_id", "paragraph_order"])
    # extract as a list of dicts
    df_dicts = df.to_dicts()
    # transpose to ids, documents, metadatas
    ids = [d["par_id"] for d in df_dicts]
    documents = [d["paragraph_text"] for d in df_dicts]
    metadatas = [{k: v for k, v in d.items() if k not in ["paragraph_text"]} for d in df_dicts]
    # metadatas = sorted(metadatas, key=lambda m: (m["doc_id"], m["paragraph_order"]))
    # linking paragraphs
    for prev, cur in zip(metadatas, metadatas[1:]):
        if prev["doc_id"] == cur["doc_id"] and prev["paragraph_order"] + 1 == cur["paragraph_order"]:
            prev["next_par_id"] = cur["par_id"]
            cur["prev_par_id"] = prev["par_id"]
        else:
            prev["next_par_id"] = ""
            cur["prev_par_id"] = ""
    return ids, documents, metadatas

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap  # slide with overlap
    return chunks

def prepare_chunks(data_filepath: str, chunk_size: int = 500, overlap: int = 50):
    ids, documents, metadatas = prepare_paragraphs(data_filepath)
    articles = {}
    for meta, text in zip(metadatas, documents):
        doc_id = meta["doc_id"]
        if doc_id not in articles:
            articles[doc_id] = {
                "text": [],
                "metadata": {k: v for k, v in meta.items() if k not in ["par_id", "paragraph_order", "prev_par_id", "next_par_id"]}
            }
        articles[doc_id]["text"].append(text)

    for doc_id in articles:
        articles[doc_id]["text"] = "\n\n".join(articles[doc_id]["text"])  # join with spacing

    chunks, chunk_metas = [], []
    for doc_id, article in articles.items():
        article_text = article["text"]
        article_meta = article["metadata"]

        text_chunks = chunk_text(article_text, chunk_size, chunk_size-overlap)

        for idx, chunk in enumerate(text_chunks):
            chunks.append(chunk)
            chunk_metas.append({
                **article_meta,
                "chunk_id": f"{doc_id}_{idx}",
                "chunk_index": idx,
                "chunk_start": idx * (chunk_size - overlap),
                "chunk_size": len(chunk),
                "total_chunks": len(text_chunks),
            })

    chunks_ids = [meta["chunk_id"] for meta in chunk_metas]

    return chunks_ids, chunks, chunk_metas


def prepare_sentences(data_filepath: str):
    # split paragraphs into sentences
    ids, documents, metadatas = prepare_paragraphs(data_filepath)
    new_ids, new_documents, new_metadatas = [], [], []
    for id_, doc, meta in zip(ids, documents, metadatas):
        sentences = [s.strip() for s in doc.split(".") if s.strip()]
        for i, sentence in enumerate(sentences):
            new_ids.append(f"s{id_}_{i}")
            new_documents.append(sentence)
            new_meta = meta.copy()
            new_meta["par_id"] = f"s{id_}_{i}"
            new_meta["sentence_order"] = i
            new_metadatas.append(new_meta)
    return new_ids, new_documents, new_metadatas


def prepare_sliding_paragraphs(data_filepath: str, window_size: int = 3, overlap: int = 2):
    ids, documents, metadatas = prepare_paragraphs(data_filepath)

    # Build fast lookup tables
    text_by_id = dict(zip(ids, documents))
    metadata_by_id = {m["par_id"]: m for m in metadatas}

    # Group paragraphs by document
    par_ids_by_doc = defaultdict(list)
    for m in metadatas:
        par_ids_by_doc[m["doc_id"]].append(m["par_id"])

    new_ids, new_documents, new_metadatas = [], [], []

    for doc_id, par_ids in par_ids_by_doc.items():
        # iterate with sliding window
        step = window_size - overlap
        for i in range(0, len(par_ids), step):
            window_par_ids = par_ids[i:i+window_size]
            if not window_par_ids:
                continue

            # build concatenated text
            window_texts = [text_by_id[pid] for pid in window_par_ids]
            new_document = "\n".join(window_texts)

            # build unique id
            new_id = "_".join(window_par_ids)

            # copy base metadata from the first paragraph in the window
            base_meta = metadata_by_id[window_par_ids[0]].copy()
            base_meta.pop("par_id", None)  # remove single par_id
            base_meta.pop("paragraph_text", None)  # remove text

            # add new fields
            base_meta["par_id"] = new_id

            new_ids.append(new_id)
            new_documents.append(new_document)
            new_metadatas.append(base_meta)

    return new_ids, new_documents, new_metadatas


def insert_documents(ids, documents, metadatas, collection, chunk=10):
    pbar = tqdm(total=len(documents), desc=f"Inserting documents ({collection.name})")
    for i in range(0, len(documents), chunk):
        collection.add(
            ids=ids[i:i+chunk],
            documents=documents[i:i+chunk],
            metadatas=metadatas[i:i+chunk]
        )
        pbar.update(chunk)
    pbar.close()


def load_data(data_filepath: str, collection_name: str, data_prepare_func, force: bool=False) -> chromadb.Collection:
    # Create the collection if it does not exist
    client = chromadb.PersistentClient(path=CHROMA_PERSIST)
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name=EMBEDDING_MODEL
    )
    em = ollama_ef
    collection = client.get_or_create_collection(name=collection_name, embedding_function=em)

    if force or collection.count() == 0:
        ids, documents, metadatas = data_prepare_func(data_filepath)
        insert_documents(ids, documents, metadatas, collection)
    return collection




if __name__ == "__main__":
    collection = load_data(DATA_FILEPATH, "paragraphes", prepare_paragraphs)
    dense_retriever = DenseRetriever(collection)
    bm25 = BM25Retriever(collection)

    dense_results = dense_retriever.retrieve("Voiture électrique", k=8)
    sparse_results = bm25.retrieve("Voiture électrique", k=8)

    if dense_results is not None:
        print("Dense results:")
        for result in dense_results:
            print("---")
            print(result)
    if sparse_results is not None:
        print("Sparse results:")
        for result in sparse_results:
            print("---")
            print(result)


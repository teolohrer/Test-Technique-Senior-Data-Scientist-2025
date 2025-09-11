from typing import List
import polars as pl
from collections import defaultdict
from tqdm import tqdm

import chromadb

from embedding import EmbeddingWrapper
from config_manager import Config


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
    # linking paragraphs
    for prev, cur in zip(metadatas, metadatas[1:]):
        if (
            prev["doc_id"] == cur["doc_id"]
            and prev["paragraph_order"] + 1 == cur["paragraph_order"]
        ):
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
                "metadata": {
                    k: v
                    for k, v in meta.items()
                    if k not in ["par_id", "paragraph_order", "prev_par_id", "next_par_id"]
                },
            }
        articles[doc_id]["text"].append(text)

    for doc_id in articles:
        articles[doc_id]["text"] = "\n\n".join(articles[doc_id]["text"])  # join with spacing

    chunks, chunk_metas = [], []
    for doc_id, article in articles.items():
        article_text = article["text"]
        article_meta = article["metadata"]

        text_chunks = chunk_text(article_text, chunk_size, overlap)

        for idx, chunk in enumerate(text_chunks):
            chunks.append(chunk)
            chunk_metas.append(
                {
                    **article_meta,
                    "chunk_id": f"{doc_id}_{idx}",
                    "chunk_index": idx,
                    "chunk_start": idx * (chunk_size - overlap),
                    "chunk_size": len(chunk),
                    "total_chunks": len(text_chunks),
                }
            )

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


def prepare_sliding_sentences(data_filepath: str, window_size: int = 3, overlap: int = 2):
    ids, documents, metadatas = prepare_sentences(data_filepath)

    # Build fast lookup tables
    text_by_id = dict(zip(ids, documents))
    metadata_by_id = {m["par_id"]: m for m in metadatas}

    # Group sentences by document
    sent_ids_by_doc = defaultdict(list)
    for m in metadatas:
        sent_ids_by_doc[m["doc_id"]].append(m["par_id"])

    new_ids, new_documents, new_metadatas = [], [], []

    for doc_id, sent_ids in sent_ids_by_doc.items():
        # iterate with sliding window
        step = window_size - overlap
        for i in range(0, len(sent_ids), step):
            window_sent_ids = sent_ids[i : i + window_size]
            if not window_sent_ids:
                continue

            # build concatenated text
            window_texts = [text_by_id[sid] for sid in window_sent_ids]
            new_document = ". ".join(window_texts) + "."

            # build unique id
            new_id = "_".join(window_sent_ids)

            # copy base metadata from the first sentence in the window
            base_meta = metadata_by_id[window_sent_ids[0]].copy()
            base_meta.pop("par_id", None)  # remove single par_id
            base_meta.pop("sentence_order", None)  # remove order

            # add new fields
            base_meta["par_id"] = new_id

            new_ids.append(new_id)
            new_documents.append(new_document)
            new_metadatas.append(base_meta)

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
            window_par_ids = par_ids[i : i + window_size]
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


class DataLoader:
    def __init__(self, model_name: str, base_ollama_url: str, chroma_persist: str):
        self.client = chromadb.PersistentClient(path=chroma_persist)

        self.embedding_function = EmbeddingWrapper(model_name=model_name, base_url=base_ollama_url)
# 
    def load_data(
        self, data_filepath: str, collection_name: str, data_prepare_func, force: bool = False
    ) -> chromadb.Collection:
        # Create the collection if it does not exist
        collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function.ef  # type: ignore
        )

        if force or collection.count() == 0:
            ids, documents, metadatas = data_prepare_func(data_filepath)
            self.insert_documents(ids, documents, metadatas, collection)
        return collection

    def insert_documents(self, ids, documents, metadatas, collection, chunk=50):
        pbar = tqdm(total=len(documents), desc=f"Inserting documents ({collection.name})")
        for i in range(0, len(documents), chunk):
            collection.add(
                ids=ids[i : i + chunk],
                documents=documents[i : i + chunk],
                metadatas=metadatas[i : i + chunk],
            )
            pbar.update(chunk)
        pbar.close()

    @staticmethod
    def from_config(config: Config):
        return DataLoader(
            model_name=config.embedding_model,
            base_ollama_url=config.ollama_base_url,
            chroma_persist=config.chroma_persist,
        )

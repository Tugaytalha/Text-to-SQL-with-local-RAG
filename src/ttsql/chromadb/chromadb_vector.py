import json
from typing import List

import chromadb
import pandas as pd
import numpy as np
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..base import VannaBase
from ..utils import deterministic_uuid


# Reranker mods class (enum)
class RerankerType:
    AutoModelForSequenceClassification = 1
    AutoTokenizer_AutoModelForSequenceClassification_Wno_grad = 2


embedding_list = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "intfloat/multilingual-e5-large-instruct",
    "jinaai/jina-embeddings-v3",
]

reranker_index = 0
reranker_list = [
    "jinaai/jina-reranker-v2-base-multilingual",
    "BAAI/bge-reranker-v2-m3",
    "Alibaba-NLP/gte-multilingual-reranker-base",
]
reranker_dict = {
    "jinaai/jina-reranker-v2-base-multilingual": RerankerType.AutoModelForSequenceClassification,
    "BAAI/bge-reranker-v2-m3": RerankerType.AutoTokenizer_AutoModelForSequenceClassification_Wno_grad,
    "Alibaba-NLP/gte-multilingual-reranker-base": RerankerType.AutoTokenizer_AutoModelForSequenceClassification_Wno_grad,
}

default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=embedding_list[2],
    trust_remote_code=True,
)

default_rf = ""


# noinspection PyTypeChecker
class ChromaDB_VectorStore(VannaBase):
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)
        if config is None:
            config = {}

        path = config.get("path", ".")
        self.embedding_function = config.get("embedding_function", default_ef)
        curr_client = config.get("client", "persistent")
        collection_metadata = config.get("collection_metadata", None)
        self.n_results_sql = config.get("n_results_sql",
                                        config.get("n_results", 10))
        self.n_results_documentation = config.get("n_results_documentation",
                                                  config.get("n_results", 10))
        self.n_results_ddl = config.get("n_results_ddl",
                                        config.get("n_results", 10))
        self.n_retrieval_ddl = config.get("n_retrieval_ddl",
                                          config.get("n_results", 50))

        if curr_client == "persistent":
            self.chroma_client = chromadb.PersistentClient(
                path=path, settings=Settings(anonymized_telemetry=False)
            )
        elif curr_client == "in-memory":
            self.chroma_client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
        elif isinstance(curr_client, chromadb.api.client.Client):
            # allow providing client directly
            self.chroma_client = curr_client
        else:
            raise ValueError(
                f"Unsupported client was set in config: {curr_client}")

        self.documentation_collection = self.chroma_client.get_or_create_collection(
            name="documentation",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )
        self.ddl_collection = self.chroma_client.get_or_create_collection(
            name="ddl",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )
        self.sql_collection = self.chroma_client.get_or_create_collection(
            name="sql",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        embedding = self.embedding_function([data])
        if len(embedding) == 1:
            return embedding[0]
        return embedding

    def get_all_embeddings(self) -> np.ndarray:
        """
        Retrieves all embeddings from all collections (DDL, Documentation, SQL).

        Returns:
            np.ndarray: A numpy array containing all embeddings, or an empty array if none exist.
        """
        all_embeddings_list = []

        try:
            # Get DDL embeddings
            ddl_data = self.ddl_collection.get(include=["embeddings"])
            if ddl_data is not None and ddl_data.get("embeddings") is not None:
                all_embeddings_list.extend(ddl_data["embeddings"])

            # Get Documentation embeddings
            doc_data = self.documentation_collection.get(
                include=["embeddings"])
            if doc_data is not None and doc_data.get("embeddings") is not None:
                all_embeddings_list.extend(doc_data["embeddings"])

            # Get SQL embeddings
            sql_data = self.sql_collection.get(include=["embeddings"])
            if sql_data is not None and sql_data.get("embeddings") is not None:
                all_embeddings_list.extend(sql_data["embeddings"])

            if all_embeddings_list is None:
                return np.empty(
                    (0, 0))  # Return empty array if no embeddings found

            return np.array(all_embeddings_list)

        except Exception as e:
            print(f"Error retrieving embeddings from ChromaDB: {e}")
            return np.empty((0, 0))  # Return empty array on error

    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
            },
            ensure_ascii=False,
        )
        id = deterministic_uuid(question_sql_json) + "-sql"
        self.sql_collection.add(
            documents=question_sql_json,
            embeddings=self.generate_embedding(question_sql_json),
            ids=id,
        )

        return id

    def add_ddl(self, ddl: str, **kwargs) -> str:
        id = deterministic_uuid(ddl) + "-ddl"
        self.ddl_collection.add(
            documents=ddl,
            embeddings=self.generate_embedding(ddl),
            ids=id,
        )
        return id

    def add_documentation(self, documentation: str, **kwargs) -> str:
        id = deterministic_uuid(documentation) + "-doc"
        self.documentation_collection.add(
            documents=documentation,
            embeddings=self.generate_embedding(documentation),
            ids=id,
        )
        return id

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        sql_data = self.sql_collection.get()

        df = pd.DataFrame()

        if sql_data is not None:
            # Extract the documents and ids
            documents = [json.loads(doc) for doc in sql_data["documents"]]
            ids = sql_data["ids"]

            # Create a DataFrame
            df_sql = pd.DataFrame(
                {
                    "id": ids,
                    "question": [doc["question"] for doc in documents],
                    "content": [doc["sql"] for doc in documents],
                }
            )

            df_sql["training_data_type"] = "sql"

            df = pd.concat([df, df_sql])

        ddl_data = self.ddl_collection.get()

        if ddl_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in ddl_data["documents"]]
            ids = ddl_data["ids"]

            # Create a DataFrame
            df_ddl = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_ddl["training_data_type"] = "ddl"

            df = pd.concat([df, df_ddl])

        doc_data = self.documentation_collection.get()

        if doc_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in doc_data["documents"]]
            ids = doc_data["ids"]

            # Create a DataFrame
            df_doc = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_doc["training_data_type"] = "documentation"

            df = pd.concat([df, df_doc])

        return df

    def remove_training_data(self, id: str, **kwargs) -> bool:
        if id.endswith("-sql"):
            self.sql_collection.delete(ids=id)
            return True
        elif id.endswith("-ddl"):
            self.ddl_collection.delete(ids=id)
            return True
        elif id.endswith("-doc"):
            self.documentation_collection.delete(ids=id)
            return True
        else:
            return False

    def remove_collection(self, collection_name: str) -> bool:
        """
        This function can reset the collection to empty state.

        Args:
            collection_name (str): sql or ddl or documentation

        Returns:
            bool: True if collection is deleted, False otherwise
        """
        if collection_name == "sql":
            self.chroma_client.delete_collection(name="sql")
            self.sql_collection = self.chroma_client.get_or_create_collection(
                name="sql", embedding_function=self.embedding_function
            )
            return True
        elif collection_name == "ddl":
            self.chroma_client.delete_collection(name="ddl")
            self.ddl_collection = self.chroma_client.get_or_create_collection(
                name="ddl", embedding_function=self.embedding_function
            )
            return True
        elif collection_name == "documentation":
            self.chroma_client.delete_collection(name="documentation")
            self.documentation_collection = self.chroma_client.get_or_create_collection(
                name="documentation",
                embedding_function=self.embedding_function
            )
            return True
        else:
            return False

    @staticmethod
    def _extract_documents(query_results) -> list:
        """
        Static method to extract the documents from the results of a query.

        Args:
            query_results (pd.DataFrame): The dataframe to use.

        Returns:
            List[str] or None: The extracted documents, or an empty list or
            single document if an error occurred.
        """
        if query_results is None:
            return []

        if "documents" in query_results:
            documents = query_results["documents"]

            if len(documents) == 1 and isinstance(documents[0], list):
                try:
                    documents = [json.loads(doc) for doc in documents[0]]
                except Exception as e:
                    return documents[0]

            return documents

    @staticmethod
    def _extract_distances(query_results) -> list:
        """
        Static method to extract the distances from the results of a query.

        Args:
            query_results (pd.DataFrame): The dataframe to use.

        Returns:
            List[float] or None: The extracted distances, or an empty list or
            single distance if an error occurred.
        """
        if query_results is None:
            return []

        if "distances" in query_results:
            distances = query_results["distances"]

            if len(distances) == 1 and isinstance(distances[0], list):
                try:
                    distances = [float(dist) for dist in distances[0]]
                except Exception as e:
                    return distances[0]

            return distances

    @staticmethod
    def sort_chunks(chunks: list, scores: list) -> list:
        """
        This function is used to sort the chunks based on the scores.

        Args:
            chunks: The chunks to sort.
            scores: The similarity scores of the chunks to the question.

        Returns:
            list: A list of chunks sorted based on the scores.
        """

        sorted_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in sorted_chunks]

    def rerank(self, question: str, chunks: list) -> list:
        """
        This function is used to rerank and filter the chunks based on the question.
        Args:
            question: The question to rerank the chunks for.
            chunks: The chunks to rerank and filter by a number.

        Returns:
            list: A list of chunks that are relevant to the question.
        """
        if len(chunks) == 0:
            return []

        scores = []

        # Score the similarity of the chunks to the question
        if reranker_dict[reranker_list[
            reranker_index]] == RerankerType.AutoModelForSequenceClassification:

            model = AutoModelForSequenceClassification.from_pretrained(
                reranker_list[reranker_index],
                torch_dtype="auto",
                trust_remote_code=True,
            )

            model.to("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()

            # Construct sentence pairs
            sentence_pairs = [[question, chunk] for chunk in chunks]

            # Compute scores
            scores = model.compute_score(sentence_pairs, max_length=1024)
        elif reranker_dict[reranker_list[reranker_index]] == RerankerType.AutoTokenizer_AutoModelForSequenceClassification_Wno_grad:

            tokenizer = AutoTokenizer.from_pretrained(
                reranker_list[reranker_index])
            model = AutoModelForSequenceClassification.from_pretrained(
                reranker_list[reranker_index],
                torch_dtype=torch.float16 if reranker_list[
                                                 reranker_index] == "Alibaba-NLP/gte-multilingual-reranker-base" else "auto",
                trust_remote_code=True,
            )
            model.eval()

            # Construct sentence pairs
            sentence_pairs = [[question, chunk] for chunk in chunks]

            with torch.no_grad():
                inputs = tokenizer(sentence_pairs, padding=True,
                                   truncation=True,
                                   return_tensors='pt', max_length=512)
                scores = model(**inputs, return_dict=True).logits.view(
                    -1, ).float()

        # Rerank the chunks based on the scores
        return self.sort_chunks(chunks, scores)

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        return ChromaDB_VectorStore._extract_documents(
            self.sql_collection.query(
                query_texts=[question],
                n_results=self.n_results_sql,
            )
        )

    def get_related_ddl(self, question: str, **kwargs) -> list:
        return ChromaDB_VectorStore._extract_documents(
            self.ddl_collection.query(
                query_texts=[question],
                n_results=kwargs.get("n_results", self.n_results_ddl),
            )
        )

    def get_related_ddl_with_score(self, question: str, **kwargs) -> pd.DataFrame:
        return self.ddl_collection.query(
                query_texts=[question],
                n_results=kwargs.get("n_results", self.n_results_ddl),
            )

    def get_related_ddl_reranked(self, question: str, **kwargs) -> list:
        chunks = ChromaDB_VectorStore._extract_documents(
            self.ddl_collection.query(
                query_texts=[question],
                n_results=self.n_retrieval_ddl,
            )
        )

        # Rerank the chunks
        reranked_chunks = self.rerank(question, chunks)

        return reranked_chunks[:self.n_results_ddl]

    def get_related_documentation(self, question: str, **kwargs) -> list:
        return ChromaDB_VectorStore._extract_documents(
            self.documentation_collection.query(
                query_texts=[question],
                n_results=self.n_results_documentation,
            )
        )

import hashlib
import os
import re
import uuid
from typing import Union
import time

from .exceptions import ImproperlyConfigured, ValidationError

import matplotlib.pyplot as plt
import umap
import numpy as np


def validate_config_path(path):
    if not os.path.exists(path):
        raise ImproperlyConfigured(
            f'No such configuration file: {path}'
        )

    if not os.path.isfile(path):
        raise ImproperlyConfigured(
            f'Config should be a file: {path}'
        )

    if not os.access(path, os.R_OK):
        raise ImproperlyConfigured(
            f'Cannot read the config file. Please grant read privileges: {path}'
        )


def sanitize_model_name(model_name):
    try:
        model_name = model_name.lower()

        # Replace spaces with a hyphen
        model_name = model_name.replace(" ", "-")

        if '-' in model_name:

            # remove double hyphones
            model_name = re.sub(r"-+", "-", model_name)
            if '_' in model_name:
                # If name contains both underscores and hyphen replace all underscores with hyphens
                model_name = re.sub(r'_', '-', model_name)

        # Remove special characters only allow underscore
        model_name = re.sub(r"[^a-zA-Z0-9-_]", "", model_name)

        # Remove hyphen or underscore if any at the last or first
        if model_name[-1] in ("-", "_"):
            model_name = model_name[:-1]
        if model_name[0] in ("-", "_"):
            model_name = model_name[1:]

        return model_name
    except Exception as e:
        raise ValidationError(e)


def deterministic_uuid(content: Union[str, bytes]) -> str:
    """Creates deterministic UUID on hash value of string or byte content.

    Args:
        content: String or byte representation of data.

    Returns:
        UUID of the content.
    """
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    elif isinstance(content, bytes):
        content_bytes = content
    else:
        raise ValueError(f"Content type {type(content)} not supported !")

    hash_object = hashlib.sha256(content_bytes)
    hash_hex = hash_object.hexdigest()
    namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
    content_uuid = str(uuid.uuid5(namespace, hash_hex))

    return content_uuid


def score_passed(score, rerank) -> bool:
    """
    Checks if the score passed the threshold
    Args:
        score: score to check
        rerank: whether it is score rerank result or distance from retrieval

    Returns:
        bool: True if the score passed the threshold
    """
    return score > 0.1 if rerank else score < 1.5


def visualize_query_embeddings(query, query_embedding, all_chunk_embeddings,
                               retrieved_embeddings):
    """
    Creates a visualization of a query and its relationship with document chunks.

    Args:
        query (str): The query text
        query_embedding (numpy.ndarray): The embedding vector of the query
        all_chunk_embeddings (numpy.ndarray): Embeddings of all document chunks
        retrieved_embeddings (numpy.ndarray): Embeddings of retrieved chunks

    Returns:
        str: Path to the saved visualization image
    """
    # Reshape query embedding if needed
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Reduce dimensions using UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')

    # Stack query and all chunk embeddings for dimensionality reduction
    all_embeddings = np.vstack((query_embedding, all_chunk_embeddings))
    reduced_embeddings = reducer.fit_transform(all_embeddings)

    # Extract 2D positions
    query_position = reduced_embeddings[0]
    all_chunk_positions = reduced_embeddings[1:]

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot all chunks in grey
    plt.scatter(all_chunk_positions[:, 0], all_chunk_positions[:, 1],
                c='grey', label='All Chunks', alpha=0.3)

    # Plot query in red
    plt.scatter(query_position[0], query_position[1],
                c='red', label='Query', s=100, edgecolor='black')

    # Plot retrieved chunks in yellow
    if len(retrieved_embeddings) > 0:
        retrieved_positions = reducer.transform(retrieved_embeddings)
        plt.scatter(retrieved_positions[:, 0], retrieved_positions[:, 1],
                    c='yellow', label='Retrieved Chunks', s=70,
                    edgecolor='black')

    # Add title and legend
    plt.title(f"Query: {query[:40]}{'...' if len(query) > 40 else ''}")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    # Create directory for visualizations if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)

    # Save to a file with a timestamp to avoid conflicts
    timestamp = int(time.time())
    output_path = f"visualizations/query_viz_{timestamp}.png"
    plt.savefig(output_path, format='png', dpi=100)
    plt.close()

    return output_path

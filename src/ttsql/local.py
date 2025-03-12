from .chromadb.chromadb_vector import ChromaDB_VectorStore
from .ollama.ollama import Ollama


class LocalContext_Ollama(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

        self._model = config["model"]

from .chromadb.chromadb_vector import ChromaDB_VectorStore
from .openai.openai_chat import OpenAI_Chat
from .ollama.ollama import Ollama


class LocalContext_OpenAI(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)


class LocalContext_Ollama(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

        self._model = config["model"]

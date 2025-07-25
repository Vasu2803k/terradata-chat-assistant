import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from functools import partial

from langchain.text_splitter import RecursiveCharacterTextSplitter

from scripts.log_config import setup_logging, get_logger

logger = get_logger(__name__)

# Set project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def is_model_cached(model_name, cache_folder):
    logger.info("---Entering is_model_cached---")
    # HuggingFace stores models in cache_folder/models--{model_name.replace('/', '--')}
    model_dir = os.path.join(cache_folder, f"models--{model_name.replace('/', '--')}")
    logger.info("---End of is_model_cached---")
    return os.path.exists(model_dir)

class TextEmbedding:
    def __init__(self, collection_name="text_embeddings", persist_directory=None, model_name="sentence-transformers/all-mpnet-base-v2", cache_folder=None):
        logger.info("---Entering TextEmbedding.__init__---")
        logger.info(f"Initializing TextEmbedding with model: {model_name}, collection: {collection_name}")
        if persist_directory is None:
            persist_directory = str(PROJECT_ROOT / "vector_db")
        if cache_folder is None:
            cache_folder = str(PROJECT_ROOT / "hf_cache")

        if not is_model_cached(model_name, cache_folder):
            logger.error(f"Model {model_name} not found in cache at {cache_folder}. Please download it manually before running.")
        else:
            logger.info(f"Model {model_name} found in cache. Loading from {cache_folder}.")
            
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, cache_folder=cache_folder)
        logger.info(f"Embedding dimension: {self.embeddings._client.get_sentence_embedding_dimension()}")
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(collection_name)
        logger.info("---End of TextEmbedding.__init__---")

    async def embed_texts(self, texts):
        logger.info("---Entering embed_texts---")
        """
        Given a list of texts, return their embeddings using HuggingFace embeddings.
        Runs blocking calls in a thread pool.
        """
        loop = asyncio.get_running_loop()
        try:
            # Use HuggingFace embeddings
            embeddings = await loop.run_in_executor(
                None,
                self.embeddings.embed_documents,
                texts
            )
            logger.info("---End of embed_texts---")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Fallback to random embeddings if API fails
            import numpy as np
            embedding_dim = 384  # all-MiniLM-L6-v2 dimension
            logger.info("---End of embed_texts---")
            return [np.random.rand(embedding_dim).tolist() for _ in texts]


    async def add_texts(self, texts, metadatas=None, ids=None):
        logger.info("---Entering add_texts---")
        """
        Embed and add texts to the Chroma vector database.
        """
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        embeddings = await self.embed_texts(texts)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(
                self.collection.add,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas if metadatas else [{} for _ in texts],
                ids=ids
            )
        )
        logger.info("---End of add_texts---")

    async def query(self, query_text, n_results=3):
        logger.info("---Entering query---")
        """
        Query the Chroma vector DB for similar texts.
        """
        query_embedding = (await self.embed_texts([query_text]))[0]
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            partial(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=n_results
            )
        )
        logger.info("---End of query---")
        return results

    def split_text_with_metadata(self, text, metadata, chunk_size=1024, chunk_overlap=100):
        logger.info("---Entering split_text_with_metadata---")
        """
        Split a large text into chunks using RecursiveCharacterTextSplitter and attach metadata to each chunk.
        Args:
            text: The full text to split
            metadata: Dict with at least 'name', 'title', and any other metadata
            chunk_size: Max size of each chunk
            chunk_overlap: Overlap between chunks
        Returns:
            (chunks, metadatas): List of text chunks and corresponding metadata dicts
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta['chunk_index'] = i
            chunk_meta['chunk_start'] = max(0, i * (chunk_size - chunk_overlap))
            chunk_meta['chunk_end'] = chunk_meta['chunk_start'] + len(chunk)
            metadatas.append(chunk_meta)
        logger.info(f"Split text into {len(chunks)} chunks with metadata: {metadata}")
        logger.info("---End of split_text_with_metadata---")
        return chunks, metadatas

    async def add_document(self, text, metadata, chunk_size=1024, chunk_overlap=100):
        logger.info("---Entering add_document---")
        """
        Split a large document, attach metadata, and add all chunks to ChromaDB.
        Args:
            text: The full text to split and embed
            metadata: Dict with at least 'name', 'title', etc
        """
        chunks, metadatas = self.split_text_with_metadata(text, metadata, chunk_size, chunk_overlap)
        await self.add_texts(chunks, metadatas=metadatas)
        logger.info(f"Added document with {len(chunks)} chunks to ChromaDB.")
        logger.info("---End of add_document---")

    async def process_files_for_embedding(self, path=None, chunk_size=1024, chunk_overlap=100):
        logger.info("---Entering process_files_for_embedding---")
        """
        Read a file or all files in a directory, split, and add to embedding DB with metadata.
        Args:
            path: file or directory path
            metadata: dict with at least 'name', 'title', etc (applied to all files, can be extended per file)
        """
    
        texts = self.read_texts_from_path(path)
        for fname, text in texts.items():
            print(f"\nProcessing file: {fname}")
            # Prompt for metadata for each file
        
            if "19AG36010_BTP-I" in fname:
                file_metadata = {"name": "Tippagudisi Deepika Sravya", "title": "Data aggregation and Analysis for Web-Based Exploration of Hydrocarbons in Prebiotic Chemistry", "filename": fname}
            elif "19HS20045_ Turre Sai Girish" in fname:
                file_metadata = {"name": "Turre Sai Girish", "title": "Unveiling the Dynamic: The Non-Neutrality Proposition of Money", "filename": fname}
            elif "19MF3IM04_BTP-II" in fname:
                file_metadata = {"name": "Vasu Katravath", "title": "Health Stages Division and Identification of Fault Occurrence Time for Rolling Element Bearings Based on Hidden Markov Model", "filename": fname}
            elif "MTP_Documentation" in fname:
                file_metadata = {"name": "Vasu Katravath", "title": "MTP Documentation", "filename": fname}
            elif "MTP_II.docx" in fname:
                file_metadata = {"name": "Vasu Katravath", "title": "Natural Language Processing Powered Maintenance Solutions for Mechanical Challenges", "filename": fname}
            elif "summer_training_repot" in fname:
                file_metadata = {"name": "Vasu Katravath", "title": "Design and Development of a Fact-Based Question Answering System Using Large Language Models", "filename": fname}
            elif "Vinod MTP Report" in fname:
                file_metadata = {"name": "Vinod MTP Report", "title": "Development of High Energy Nutrient-Dense Cookies for Undernourished Adolescents", "filename": fname}
            else:
                file_metadata = {"name": "Unknown", "title": "Unknown", "filename": fname}

            logger.info(f"Processing file {fname} for embedding with metadata: {file_metadata}")
            await self.add_document(text, file_metadata, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info(f"Processed {len(texts)} file(s) for embedding from path: {path}")
        logger.info("---End of process_files_for_embedding---")

    def read_texts_from_path(self, path):
        logger.info("---Entering read_texts_from_path---")
        """
        Read text from a file or all files in a directory.
        Args:
            path: Path to a file or directory
        Returns:
            If file: (filename, text)
            If dir: dict {filename: text}
        """
        import os
        texts = {}
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"Read text from file: {path}")
            logger.info("---End of read_texts_from_path---")
            return {os.path.basename(path): text}
        elif os.path.isdir(path):
            for fname in os.listdir(path):
                fpath = os.path.join(path, fname)
                if os.path.isfile(fpath):
                    with open(fpath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    texts[fname] = text
                    logger.info(f"Read text from file: {fpath}")
            logger.info("---End of read_texts_from_path---")
            return texts
        else:
            logger.error(f"Path does not exist: {path}")
            logger.info("---End of read_texts_from_path---")
            return {}

def text_embedding_tool(*args, **kwargs):
    print("=== Text Embedding Tool ===")
    path=str(PROJECT_ROOT / "output")
    
    text_embedder = TextEmbedding()
    asyncio.run(text_embedder.process_files_for_embedding(path))
    print("Embedding complete.")

if __name__=="__main__":
    text_embedding_tool()
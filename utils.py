from pydantic import BaseModel, Field
from enum import Enum
import json, asyncio
import numpy as np
import requests
from pathlib import Path
import hashlib
import faiss
from tqdm import tqdm
from constants import *
from enum import Enum
from document_processor import *
from logging import Logger
from sub_prompts import FunctionName

system_prompt = """
You are a RAG Agent skilled at answering questions related to the field of sports (Especially in the cricketing world)
"""


class FunctionCall(BaseModel):
    what_was_done_in_previous_step: str = Field(
        ..., description="A brief of what action was performed in the previous step"
    )
    what_needs_to_be_done_next: str = Field(
        ...,
        description="What is the next step that needs to be done as per the plan of action to complete the user's task",
    )
    tool_name: FunctionName
    arguments: dict | None


async def generate_with_timeout(client, prompt, timeout=60):
    """
    Generate content with a timeout using the OpenAI API.

    This function:
    1. Makes an asynchronous call to the OpenAI API
    2. Handles timeouts and errors
    3. Returns the generated content

    Args:
        client: OpenAI client instance
        prompt (str): The prompt to generate content for
        timeout (int): Maximum time to wait for generation in seconds

    Returns:
        str: The generated content

    Raises:
        TimeoutError: If generation takes longer than timeout
        Exception: If there's an error in the generation process
    """
    print("Starting LLM generation...")
    try:
        # Convert the synchronous chat.completions.create call to run in a thread
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                ),
            ),
            timeout=timeout,
        )
        print("LLM generation completed")
        return response.choices[0].message.content
    except TimeoutError:
        print("LLM generation timed out!")
        raise
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        raise


def get_description_from_tools(tools):
    """
    Generate a formatted description of available tools.

    This function:
    1. Iterates through the list of tools
    2. Formats each tool's name and description
    3. Combines them into a single string

    Args:
        tools (list): List of tool objects

    Returns:
        str: Formatted description of all tools

    Raises:
        Exception: If there's an error processing the tools
    """
    try:
        tools_description = []
        for i, tool in enumerate(tools):
            try:
                tool_desc = f"{i+1}. {tool.name}\n{tool.description}"
                tools_description.append(tool_desc)
            except Exception as e:
                print(f"Error processing tool {i}: {e}")
                tools_description.append(f"{i+1}. Error processing tool")

        tools_description = "\n".join(tools_description)
        print("Successfully created tools description")
    except Exception as e:
        print(f"Error creating tools description: {e}")
        tools_description = "Error loading tools"

    return tools_description


def get_embedding(text: str, url: str, model: str) -> np.ndarray:
    """
    Generate an embedding vector for the given text using the specified model.

    Args:
        text (str): The text to embed
        url (str): URL of the embedding service
        model (str): Name of the embedding model to use

    Returns:
        np.ndarray: The embedding vector

    Raises:
        requests.exceptions.RequestException: If the embedding request fails
    """
    response = requests.post(url, json={"model": model, "prompt": text})
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)


def get_file_hash(path):
    """
    Generate a hash value for a file's contents.

    Args:
        path (str): Path to the file

    Returns:
        str: MD5 hash of the file's contents
    """
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def process_documents(logger: Logger = None):
    """
    Process documents and create a FAISS index for efficient similarity search.

    This function:
    1. Processes documents using either semantic or naive chunking
    2. Generates embeddings for each chunk
    3. Creates and updates the FAISS index
    4. Stores metadata and cache information

    Args:
        logger (Logger, optional): Logger instance for logging operations
    """
    INDEX_CACHE.mkdir(exist_ok=True)

    # Define paths for FAISS
    INDEX_FILE = INDEX_CACHE / "index.bin"
    METADATA_FILE = INDEX_CACHE / "metadata.json"
    CACHE_FILE = INDEX_CACHE / "doc_index_cache.json"

    cache_meta = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
    index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else None

    if CHUNKER == "semantic":
        chunker = SemanticDocChunker(block_size=SEMANTIC_CHUNKING_BLOCK_SIZE)
        if logger:
            logger.info(f"Using semantic chunking strategy")
    else:
        chunker = NaiveChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        if logger:
            logger.info(f"Using naive chunking strategy")

    for file in list(DOC_PATH.glob("*")):
        hsh = get_file_hash(file)

        # If the file was previously hashed and the contents of the files have not at all changed
        if (file.name in cache_meta) and cache_meta[file.name] == hsh:
            continue
        try:
            chunks = chunker.chunk_doc(str(file), logger)

            embeddings_for_file = []
            new_metadata = []

            for idx, chunk in enumerate(
                tqdm(chunks, desc=f"Creating embeddings"), start=1
            ):
                embedding = get_embedding(chunk, OLLAMA_URL, EMBEDING_MODEL_NAME)
                embeddings_for_file.append(embedding)
                new_metadata.append(
                    {"doc": file.name, "chunk": chunk, "chunk_id": f"{file.name}_{idx}"}
                )

            if len(embeddings_for_file) > 0:
                if index is None:
                    embed_dimension = len(embeddings_for_file[0])
                    index = faiss.IndexFlatL2(embed_dimension)
                index.add(np.stack(embeddings_for_file))
                metadata.extend(new_metadata)
            cache_meta[file.name] = hsh
        except Exception as e:
            print(str(e))

    CACHE_FILE.write_text(json.dumps(cache_meta, indent=2))
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))

    if index and index.ntotal > 0:
        faiss.write_index(index, str(INDEX_FILE))


def ensure_faiss_ready():
    """
    Ensure the FAISS index is ready for use.

    This function:
    1. Checks if the index and metadata files exist
    2. If not, processes documents to create them
    3. Logs the status of the index
    """
    index_path = INDEX_CACHE / "index.bin"
    meta_path = INDEX_CACHE / "metadata.json"
    if not (index_path.exists() and meta_path.exists()):
        print("INFO", "Index not found — running process_documents()...")
        process_documents()
    else:
        print("INFO", "Index already exists. Skipping regeneration.")


if __name__ == "__main__":
    from logging import Logger
    from rich.logging import RichHandler
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = logging.getLogger("utils")
    process_documents(logger)

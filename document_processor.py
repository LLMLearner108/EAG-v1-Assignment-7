from constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)
from pathlib import Path
from sub_prompts import chunking_prompt_instruction
import requests
from logging import Logger
from markitdown import MarkItDown
from openai import OpenAI
from dotenv import load_dotenv
import os


class NaiveChunker:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.md = MarkItDown()

    def _process_doc(self, doc_path) -> str:
        result = self.md.convert(doc_path)
        return result.text_content

    def chunk_doc(self, doc_path: str, logger: Logger = None):
        doc_path = Path(doc_path)
        text = self._process_doc(doc_path)
        words = text.split(" ")
        chunks = []

        if logger:
            logger.info(f"Document: {doc_path.stem}")
            logger.info(f"Number of words in the document: {len(words)}")

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunks.append(" ".join(words[i : i + self.chunk_size]))

        if logger:
            logger.info(f"Broke the document into {len(chunks)} chunks.")

        return chunks


class SemanticDocChunker:
    def __init__(self, block_size: int = 512):
        self.word_block_size = block_size
        self.md = MarkItDown()
        load_dotenv()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _process_doc(self, doc_path) -> str:
        result = self.md.convert(doc_path)
        return result.text_content

    def chunk_doc(self, doc_path: str, logger: Logger = None):
        doc_path = Path(doc_path)
        text = self._process_doc(doc_path)
        words = text.split(" ")

        if logger:
            logger.info(f"Document:{doc_path.stem}")
            logger.info(f"Number of words in the document: {len(words)}")

        word_index = 0
        chunks = []

        while word_index < len(words):
            current_chunks = words[word_index : word_index + self.word_block_size]
            chunk_text = " ".join(current_chunks).strip()

            prompt = chunking_prompt_instruction.format(chunk_text=chunk_text)

            try:
                response = (
                    self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=128,
                    ),
                )

                reply = response[0].choices[0].message.content

                if reply:
                    # Figure out where the second part begins
                    second_part_idx = chunk_text.find(reply)

                    # If the second part is not towards the end
                    if second_part_idx != -1:
                        ready_chunk = chunk_text[:second_part_idx].strip()
                        spillover_chunk = chunk_text[second_part_idx:]
                        chunks.append(ready_chunk)
                        spillover_words = spillover_chunk.split(" ")

                        # Overwrite the words to remove words corresponding to the ready chunk
                        words = (
                            spillover_words + words[word_index + self.word_block_size :]
                        )
                        word_index = 0
                        continue
                    else:
                        chunks.append(chunk_text)
                else:
                    chunks.append(chunk_text)
            except Exception as e:
                if logger:
                    logger.error(
                        f"Error when parsing chunk: {chunk_text}\nError: {str(e)}"
                    )

            word_index += self.word_block_size

        if logger:
            logger.info(f"Broke the document into {len(chunks)} chunks.")

        return chunks

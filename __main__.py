import logging

import gradio as gr
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from ollama import Client
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)


def ollama_chat(messages, model="llama3.2"):
    return OLLAMA_CLIENT.chat(
        model=model,
        messages=messages,
    ).message.content


def qdrant_store(documents, metadata):
    QDRANT_CLIENT.add(
        collection_name=COLLECTION_NAME,
        documents=documents,
        metadata=metadata,
        batch_size=64,
    )
    logging.info(f"New document added to collection: {documents, metadata}")


def qdrant_query(query_text, limit=10):
    results = QDRANT_CLIENT.query(
        collection_name=COLLECTION_NAME, query_text=query_text, limit=limit
    )
    logging.info(f"Results found for the query '{query_text}':\n{results}")
    return results


def gradio_chat(message, history):
    for key, value in message.items():
        if key == "text":
            question = value
        if key == "files":
            for item in value:
                file = DOC_CONVERTER.convert(item)
                documents, metadatas = [], []
                for chunk in HybridChunker().chunk(file.document):
                    documents.append(chunk.text)
                    metadatas.append(chunk.meta.export_json_dict())
                qdrant_store(documents, metadatas)

    # Get Context
    points = qdrant_query(question)
    for index, point in enumerate(points):
        chat_history.append(
            {"role": "user", "content": f"Context {index + 1}: {point}"}
        )

    # Create Question
    user = f"Question: '{question}'."
    chat_history.append({"role": "user", "content": user})

    # Get Response
    response = ollama_chat(chat_history)
    logging.info(f"Response: {response}")
    chat_history.append({"role": "assistant", "content": response})

    return response


if __name__ == "__main__":
    # Initialize Ollama and Qdrant Clients
    OLLAMA_CLIENT = Client()
    QDRANT_CLIENT = QdrantClient()
    QDRANT_CLIENT.set_model("sentence-transformers/all-MiniLM-L6-v2")
    QDRANT_CLIENT.set_sparse_model("Qdrant/bm25")
    DOC_CONVERTER = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.CSV,
            InputFormat.MD,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline, backend=PyPdfiumDocumentBackend
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline, backend=MsWordDocumentBackend
            ),
        },
    )

    # Create Qdrant Collection
    COLLECTION_NAME = "docling"
    if QDRANT_CLIENT.collection_exists(COLLECTION_NAME):
        QDRANT_CLIENT.delete_collection(COLLECTION_NAME)

    if not QDRANT_CLIENT.collection_exists(COLLECTION_NAME):
        QDRANT_CLIENT.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=QDRANT_CLIENT.get_fastembed_vector_params(),
            sparse_vectors_config=QDRANT_CLIENT.get_fastembed_sparse_vector_params(),
        )

    # Enrich Chat History
    chat_history = []
    system = """I am going to ask you a question, which I would like you to answer
        based only on the provided context, and not any other information.
        If there is not enough information in the context to answer the question,
        say "I am not sure", then try to make a guess.
        Break your answer up into nicely readable paragraphs."""
    chat_history.append({"role": "system", "content": system})

    gr.ChatInterface(
        fn=gradio_chat,
        type="messages",
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            file_count="single",
            file_types=[".pdf", ".docx", ".html", ".pptx", ".asciidoc", ".csv", ".md"],
        ),
        save_history=True,
        theme="base",
    ).launch()

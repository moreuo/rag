import logging
import tempfile

import chromadb
import gradio as gr
from chromadb.config import Settings
from git import Repo
from ollama import Client

logging.basicConfig(level=logging.INFO)


def create_response(message, model="deepseek-r1"):
    client = Client()

    return client.chat(
        model=model,
        messages=message,
    ).message.content


def clone_repository(repo_url):
    temp_dir = tempfile.mkdtemp()
    repo = Repo.clone_from(repo_url, temp_dir)
    logging.info(f"{repo_url} cloned to {temp_dir}.")
    return repo


def store_in_chromadb(collection, data):
    count = 0
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
    for docs in data:
        if docs["ids"] not in existing_ids:
            count += 1
            collection.add(ids=docs["ids"], documents=docs["documents"])
    logging.info("No new documents to insert.") if count == 0 else logging.info(
        f"{count} new documents added to collection."
    )


def query_chromadb(collection, query, n_results=2):
    results = collection.query(query_texts=[query], n_results=n_results)
    logging.info(f"Results found for the query '{query}':\n{results}")
    return results


def extract_data(repo):
    extracted_data = []
    for commit in repo.iter_commits():
        commit_data = {
            "ids": commit.hexsha,
            "documents": str(
                {
                    "author": commit.author.name,
                    "email": commit.author.email,
                    "datetime": commit.committed_datetime.isoformat(),
                    "message": commit.message,
                    "diff": ", ".join([diff.a_path for diff in commit.diff(None)]),
                    "branch": repo.active_branch.name
                    if not repo.head.is_detached
                    else "detached",
                }
            ),
        }
        extracted_data.append(commit_data)

    logging.info(f"{len(extracted_data)} extracted commits from the repository.")
    return extracted_data


def chat_interface(message, history):
    # Query Collection
    context = query_chromadb(collection, message)

    # Ask Question
    user = f"The question is '{message}'. Here is all the context you have: {context}"
    chat_history.append({"role": "user", "content": user})

    # Create Response
    response = create_response(chat_history)
    logging.info(f"Response: {response}")
    chat_history.append({"role": "system", "content": response})

    return response


if __name__ == "__main__":
    # Extract Repository Data
    url = ""
    assert url, "Please provide a repository url."
    repo = clone_repository(url)

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(
        path="chroma.db", settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_or_create_collection(name="repository")

    # Load and Insert Documents
    documents = extract_data(repo)
    store_in_chromadb(collection, documents)

    # Enrich Chat History
    chat_history = []
    assistant = """I am going to ask you a question, which I would like you to answer
        based only on the provided context, and not any other information.
        If there is not enough information in the context to answer the question,
        say "I am not sure", then try to make a guess.
        Break your answer up into nicely readable paragraphs."""
    chat_history.append({"role": "system", "content": assistant})

    gr.ChatInterface(
        fn=chat_interface, type="messages", title="Chatbot", save_history=True
    ).launch()

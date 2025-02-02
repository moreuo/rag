import tempfile

import chromadb
from git import Repo
from ollama import Client


def create_response(message, model="deepseek-r1"):
    client = Client()

    return client.chat(
        model=model,
        messages=message,
    ).message.content


def insert_documents(collection, repo_data):
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

    new_docs = [
        {
            "id": commit["commit_hash"],
            "document": commit["message"],
            "metadata": {
                "author": commit["author"],
                "email": commit["email"],
                "datetime": commit["datetime"],
                "changed_files": ", ".join(
                    [
                        f"{file['file']} ({file['change_type']})"
                        for file in commit["changed_files"]
                    ]
                ),
            },
        }
        for commit in repo_data["documents"]
        if commit["commit_hash"] not in existing_ids
    ]

    if new_docs:
        collection.add(
            ids=[doc["id"] for doc in new_docs],
            documents=[doc["document"] for doc in new_docs],
            metadatas=[doc["metadata"] for doc in new_docs],
        )
        print(f"* {len(new_docs)} new commits inserted!")
    else:
        print("* No new commits to insert!")


def query_collection(collection, query, n_results=5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    if not results["metadatas"]:
        return None

    return [
        {
            "id": results["ids"][i][0],
            "metadata": results["metadatas"][i][0],
            "document": results["documents"][i][0] if results["documents"] else None,
        }
        for i in range(len(results["metadatas"]))
    ]


def extract_repo_data(repo_url):
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = Repo.clone_from(repo_url, temp_dir)

        repo_info = {
            "repo_name": repo_url.split("/")[-1],
            "branches": [branch.name for branch in repo.branches],
            "tags": [
                {"name": tag.name, "commit": tag.commit.hexsha} for tag in repo.tags
            ],
        }

        commits_data = []
        for commit in repo.iter_commits():
            commits_data.append(
                {
                    "commit_hash": commit.hexsha,
                    "author": commit.author.name,
                    "email": commit.author.email,
                    "message": commit.message.strip(),
                    "datetime": commit.committed_datetime.isoformat(),
                    "changed_files": [
                        {"file": diff.a_path, "change_type": diff.change_type}
                        for diff in commit.diff(commit.parents[0])
                    ]
                    if commit.parents
                    else [],
                }
            )

        data = {
            "id": repo_info["repo_name"],
            "metadata": {
                "repo_info": repo_info,
                "commit_count": len(commits_data),
            },
            "documents": commits_data,
        }

        return data


if __name__ == "__main__":
    chat_history = []

    # Chroma Collection
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="repository")

    # Extract Repository Data
    repository_url = ""

    if collection.count() == 0:
        print("* Extracting repository data...")
        data = extract_repo_data(repository_url)
        print("* Inserting into the database...")
        insert_documents(collection, data)

    # Chat Loop
    while True:
        query = input("=> Ask Me: ")

        if query.strip() == "":
            print("\nGoodbye!")
            break

        # Query Collection
        relevant_data = query_collection(collection, query)

        if not relevant_data:
            print("\n* Error: No relevant information in the database. *\n")
            continue

        # Enrich Chat History
        context = (
            "To answer the user question, you must use the relevant data below:\n\n"
        )
        for data in relevant_data:
            context += (
                f"Commit: {data['id']}\n"
                f"Author: {data['metadata']['author']}\n"
                f"Date: {data['metadata']['datetime']}\n"
                f"Message: {data['document']}\n"
                f"Changed Files: {data['metadata']['changed_files']}\n\n"
            )
        chat_history.append({"role": "assistant", "content": context})
        chat_history.append({"role": "user", "content": query})

        # Create Response
        response = create_response(chat_history)
        print("\n=> Response:", response, "\n")
        chat_history.append({"role": "assistant", "content": response})

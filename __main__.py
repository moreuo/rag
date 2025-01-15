import chromadb
from ollama import Client

from database import github_faq


def insert_documents(collection, docs):
    if collection.count() == 0:
        for item in docs:
            collection.add(
                documents=[item["question"] + " " + item["answer"]],
                metadatas=[{"question": item["question"], "answer": item["answer"]}],
                ids=[item["question"]],
            )


def query_collection(collection, query):
    results = collection.query(
        query_texts=[query],
        n_results=1,
    )
    return results["metadatas"][0] if results["metadatas"] else None


def create_response(message):
    client = Client()

    return client.chat(
        model="llama3.2",
        messages=message,
    ).message.content


if __name__ == "__main__":
    chat_history = []

    # Chroma Collection
    chroma_client = chromadb.Client()
    faq_collection = chroma_client.get_or_create_collection(
        name="github_faq",
    )

    # Populate Collection
    insert_documents(faq_collection, github_faq)

    # Chat Loop
    while True:
        query = input("=> Ask Me: ")

        if query == "":
            print("\nGoodbye!")
            break

        # Query Collection
        relevant_data = query_collection(faq_collection, query)
        if not relevant_data:
            print("\n* Error: No relevant information in the database. *\n")
            continue

        # Enrich Chat History
        context = (
            "To answer the user question, you must use the relevant FAQs below:\n\n"
        )
        for data in relevant_data:
            context += f"Question: {data['question']}\nAnswer: {data['answer']}\n\n"
        chat_history.append({"role": "assistant", "content": context})
        chat_history.append({"role": "user", "content": query})

        # Create Response
        response = create_response(chat_history)
        print("\n=> Response:", response, "\n")
        chat_history.append({"role": "assistant", "content": response})

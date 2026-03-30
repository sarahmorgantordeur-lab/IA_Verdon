from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL  = "llama3.2"

# --- Chargement et indexation des documents ---

loader = TextLoader("docs.txt", encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# --- Prompt personnalisé ---

prompt = PromptTemplate.from_template("""
Tu es un assistant professionnel pour notre entreprise.
Réponds uniquement à partir du contexte fourni.
Si l'information n'est pas disponible, réponds : "Je n'ai pas cette information dans ma base de connaissances."
Sois concis et précis.

Contexte :
{context}

Question :
{question}

Réponse :
""")

llm = ChatOllama(model=CHAT_MODEL, temperature=0)

# --- Chaîne RAG (LCEL) ---

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Boucle interactive ---

if __name__ == "__main__":
    print(f"Chatbot entreprise ({CHAT_MODEL}) — tapez 'quitter' pour arrêter.\n")
    while True:
        question = input("Vous : ").strip()
        if not question:
            continue
        if question.lower() in ("quitter", "exit", "quit"):
            print("Au revoir !")
            break
        reponse = chain.invoke(question)
        print(f"Bot : {reponse}\n")

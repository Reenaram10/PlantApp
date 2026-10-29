import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class HistoryFAISSMetaEngine:
    """
    A FAISS-backed RAG engine for retrieving semantic chat history.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", index_path="local_history_faiss_index"):
        print(f" [History FAISS] Initializing embeddings with {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_db = None
        self.index_path = index_path
        
        # Load or create
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            # Create an empty FAISS index by creating a dummy record
            print(" [History FAISS] Creating new blank FAISS index for chat history.")
            doc = Document(page_content="System Initialized.", metadata={"user_id": "system"})
            self.vector_db = FAISS.from_documents([doc], self.embeddings)
            self.vector_db.save_local(self.index_path)

    def load_index(self):
        """Load index from disk if it exists"""
        if os.path.exists(self.index_path):
            try:
                self.vector_db = FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print(" [History FAISS] Loaded history index from disk.")
                return True
            except Exception as e:
                print(f" [History FAISS] Error loading history index: {e}")
                return False
        return False

    def add_interaction(self, user_id, query, response):
        """Add a conversation turn to the FAISS index."""
        if not self.vector_db:
            return

        content = f"User asked: {query}\nAI Responded: {response}"
        doc = Document(
            page_content=content,
            metadata={"user_id": str(user_id)} if user_id else {"user_id": "anonymous"}
        )
        
        self.vector_db.add_documents([doc])
        self.vector_db.save_local(self.index_path)

    def get_semantic_history(self, user_id, current_query, k=3):
        """Retrieve most similar prior discussion turns for a user."""
        if not self.vector_db:
            return ""
            
        try:
            # We strictly filter by user ID so users don't see each other's memories!
            target_user = str(user_id) if user_id else "anonymous"
            search_filter = {"user_id": target_user}
                
            results = self.vector_db.similarity_search(
                current_query, 
                k=k, 
                filter=search_filter
            )
            
            if not results:
                return ""
                
            formatted = []
            for doc in results:
                formatted.append(doc.page_content)
                
            history_text = "Retrieved Semantic History Context (Relevant past messages from this user):\n" + "\n---\n".join(formatted) + "\n\n"
            return history_text
        except Exception as e:
            print(f" [History FAISS] Error during retrieval: {e}")
            return ""

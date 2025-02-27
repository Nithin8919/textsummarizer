import logging
import os
from transformers import AutoTokenizer, pipeline
import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SuperSummarizer:
    def __init__(self, legal_docs_path="legal_documents/"):
        self.tokenizer = None
        self.summarizer = None
        self.retriever = None
        self.index = None
        self.documents = []
        self.legal_docs_path = legal_docs_path  # Path to legal documents
        self._load_pipeline()
        self._load_retriever()

    def _load_pipeline(self):
        """Load the summarization pipeline."""
        try:
            model_name = "sshleifer/distilbart-cnn-6-6"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                tokenizer=self.tokenizer,
                device=-1
            )
            logging.info(f"Loaded summarization pipeline with model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading pipeline: {e}")
            raise

    def _load_retriever(self):
        """Load FAISS retriever with legal documents."""
        try:
            self.retriever = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
            self.documents = self._load_documents()  # Load legal documents

            if not self.documents:
                logging.warning("No legal documents found. RAG retrieval will be limited.")
                return
            
            embeddings = self.retriever.encode(self.documents, normalize_embeddings=True)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Cosine similarity
            self.index.add(embeddings)
            logging.info(f"Loaded {len(self.documents)} legal documents into FAISS index")
        except Exception as e:
            logging.error(f"Error loading retriever: {e}")
            raise

    def _load_documents(self):
        """Load all legal documents from a folder."""
        docs = []
        if not os.path.exists(self.legal_docs_path):
            logging.warning(f"Legal documents folder '{self.legal_docs_path}' not found.")
            return docs

        for filename in os.listdir(self.legal_docs_path):
            if filename.endswith(".txt"):  
                with open(os.path.join(self.legal_docs_path, filename), "r", encoding="utf-8") as f:
                    docs.append(f.read().strip())

        logging.info(f"Loaded {len(docs)} legal documents.")
        return docs

    def retrieve_context(self, text, k=2):
        """Retrieve relevant legal documents using FAISS."""
        try:
            if not self.documents:
                logging.warning("No legal documents indexed. Retrieval skipped.")
                return ""
            
            query_embedding = self.retriever.encode([text], normalize_embeddings=True)
            distances, indices = self.index.search(query_embedding, k)
            retrieved_docs = [self.documents[idx] for idx in indices[0]]
            return " ".join(retrieved_docs)
        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            return ""

    def web_search(self, query, max_chars=1000):
        """Web search for additional context."""
        try:
            search_url = f"https://www.google.com/search?q={query}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            snippets = soup.find_all("span")
            context = " ".join([s.get_text() for s in snippets[:5]])[:max_chars]
            return context
        except Exception as e:
            logging.error(f"Error in web search: {e}")
            return ""

    def summarize(self, text, max_input_length=1024, max_summary_length=256, min_summary_length=50, use_rag=False):
        """Summarize input text with optional RAG enhancement."""
        try:
            if not text or not isinstance(text, str):
                return "Error: Please enter a non-empty text string."

            tokens = self.tokenizer(text, truncation=True, max_length=max_input_length, return_tensors="pt")
            truncated_text = self.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

            # RAG enhancement
            if use_rag:
                context = self.retrieve_context(truncated_text)
                if not context:
                    context = self.web_search(truncated_text[:50])  # Use first 50 chars as query
                enhanced_input = f"{truncated_text} Additional context: {context}"
                logging.info("Using RAG with retrieved context")
            else:
                enhanced_input = truncated_text

            gen_kwargs = {
                "length_penalty": 1.0,
                "num_beams": 6,
                "max_length": max_summary_length,
                "min_length": min_summary_length,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            }

            logging.info("Generating summary...")
            summary = self.summarizer(enhanced_input, **gen_kwargs)[0]["summary_text"]
            logging.info("Summary generated successfully.")
            return summary
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return f"Error: {str(e)}"

# Initialize summarizer
summarizer = SuperSummarizer()

# Gradio interface function
def summarize_text(input_text, summary_length, use_rag):
    max_len = int(summary_length)
    return summarizer.summarize(input_text, max_summary_length=max_len, min_summary_length=max_len//2, use_rag=use_rag)

# Create Gradio interface
interface = gr.Interface(
    fn=summarize_text,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter your text here...", label="Input Text"),
        gr.Slider(minimum=128, maximum=512, step=64, value=256, label="Summary Length (tokens)"),
        gr.Checkbox(label="Enhance with RAG (Retrieval-Augmented Generation)", value=False)
    ],
    outputs=gr.Textbox(label="Super Summary"),
    title="Text Summarization using Nlp and Transformers",
    description="Summarize text with optional RAG enhancement for richer context. Works with legal documents!",
    theme="huggingface"
)

if __name__ == "__main__":
    interface.launch(share=True)

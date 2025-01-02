from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def grammar_correction(text):
    doc = nlp(text)
    corrected_text = " ".join([token.text for token in doc])
    return corrected_text

def remove_redundancy(summary):
    sentences = summary.split(". ")
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)

    selected_sentences = []
    for i, sentence in enumerate(sentences):
        if not any(similarity_matrix[i][j] > 0.8 for j in range(i)):
            selected_sentences.append(sentence)

    return ". ".join(selected_sentences)

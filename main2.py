import logging
from transformers import AutoTokenizer, pipeline
import gradio as gr

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LightweightSummarizer:
    def __init__(self):
        self.tokenizer = None
        self.summarizer = None
        self._load_pipeline()

    def _load_pipeline(self):
        """Load a lightweight summarization model and tokenizer."""
        try:
            model_name = "sshleifer/distilbart-cnn-6-6"  # Replace if needed
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                tokenizer=self.tokenizer,
                device=-1  # Use CPU
            )
            logging.info(f"Loaded summarization pipeline with model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading pipeline: {e}")
            raise

    def summarize(self, text, max_input_length=512):
        """Generate a summary for the given text."""
        try:
            if not text or not isinstance(text, str):
                return "Error: Please enter a non-empty text string."
            tokens = self.tokenizer(text, truncation=True, max_length=max_input_length, return_tensors="pt")
            truncated_text = self.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            gen_kwargs = {"length_penalty": 0.8, "num_beams": 4, "max_length": 128, "min_length": 30}
            logging.info("Generating summary...")
            summary = self.summarizer(truncated_text, **gen_kwargs)[0]["summary_text"]
            logging.info("Summary generated successfully.")
            return summary
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return f"Error: {str(e)}"

# Initialize the summarizer
summarizer = LightweightSummarizer()

# Define the Gradio interface function
def summarize_text(input_text):
    return summarizer.summarize(input_text)

# Create the Gradio interface
interface = gr.Interface(
    fn=summarize_text,  # Function to call
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter your text here...",
        label="Input Text"
    ),  # Input component
    outputs=gr.Textbox(label="Summary"),  # Output component
    title="Text Summarization using Nlp and Transformers",
    description="Enter text to generate a concise summary using a lightweight model.",
    theme="default"  # Optional: change to "huggingface" or "soft" for different looks
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
import logging
from transformers import AutoTokenizer, pipeline
import gradio as gr
from src.textSummarizer.pipeline.predicition_pipeline import PredictionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LightweightSummarizer:
    def __init__(self):
        self.tokenizer = None
        self.summarizer = None
        self._load_pipeline()

    def _load_pipeline(self):
        try:
            model_name = "sshleifer/distilbart-cnn-6-6"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                tokenizer=self.tokenizer,
                device=-1  # CPU
            )
            logging.info(f"Loaded summarization pipeline with model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading pipeline: {e}")
            raise

    def summarize(self, text, max_input_length=1024, max_summary_length=256, min_summary_length=50):
        try:
            if not text or not isinstance(text, str):
                return "Error: Please enter a non-empty text string."
            
            # Tokenize and truncate input
            tokens = self.tokenizer(text, truncation=True, max_length=max_input_length, return_tensors="pt")
            truncated_text = self.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            
            # Adjusted generation settings for longer, better summaries
            gen_kwargs = {
                "length_penalty": 1.0,        # Neutral (1.0) or encourage longer (e.g., 2.0)
                "num_beams": 6,              # More beams for better quality
                "max_length": max_summary_length,  # Increase for longer summaries
                "min_length": min_summary_length,  # Ensure substantial output
                "no_repeat_ngram_size": 3,   # Prevent repetition
                "early_stopping": True       # Stop when beams converge
            }

            logging.info("Generating summary...")
            summary = self.summarizer(truncated_text, **gen_kwargs)[0]["summary_text"]
            logging.info("Summary generated successfully.")
            return summary
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return f"Error: {str(e)}"

# Initialize summarizer
summarizer = LightweightSummarizer()

# Gradio interface function with length slider
def summarize_text(input_text, summary_length):
    max_len = int(summary_length)  # User-selected length (e.g., 128, 256, 512)
    return summarizer.summarize(input_text, max_summary_length=max_len, min_summary_length=max_len//2)

# Create Gradio interface
interface = gr.Interface(
    fn=summarize_text,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter your text here...", label="Input Text"),
        gr.Slider(minimum=128, maximum=512, step=64, value=256, label="Summary Length (tokens)")
    ],
    outputs=gr.Textbox(label="Summary"),
    title="Text summarization using NLp and Transformers",
    description="Enter text and adjust the slider to control summary length.",
    theme="default"
)

if __name__ == "__main__":
    interface.launch(share = True)
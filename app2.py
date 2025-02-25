import logging
from transformers import AutoTokenizer, pipeline
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
                device=-1
            )
            logging.info(f"Loaded summarization pipeline with model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading pipeline: {e}")
            raise

    def summarize(self, text, max_input_length=512):
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Input text must be a non-empty string.")
            tokens = self.tokenizer(text, truncation=True, max_length=max_input_length, return_tensors="pt")
            truncated_text = self.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            gen_kwargs = {"length_penalty": 0.8, "num_beams": 4, "max_length": 128, "min_length": 30}
            logging.info("Generating summary...")
            summary = self.summarizer(truncated_text, **gen_kwargs)[0]["summary_text"]
            logging.info("Summary generated successfully.")
            return summary
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            raise

def main():
    summarizer = LightweightSummarizer()
    print("Lightweight Text Summarizer")
    print("Enter your text below (type 'quit' to exit):")
    
    while True:
        text = input("> ")
        if text.lower() == "quit":
            print("Exiting...")
            break
        try:
            summary = summarizer.summarize(text)
            print("\nSummary:")
            print(summary)
            print("\nEnter more text or 'quit' to exit:")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
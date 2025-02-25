import logging
from transformers import AutoTokenizer, pipeline
import gradio as gr

# Logging setup
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

    def summarize(self, text, max_input_length=1024, max_summary_length=256, min_summary_length=50, length_penalty=1.0):
        try:
            if not text or not isinstance(text, str):
                return "Error: Please enter a non-empty text string."
            tokens = self.tokenizer(text, truncation=True, max_length=max_input_length, return_tensors="pt")
            truncated_text = self.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            gen_kwargs = {
                "length_penalty": length_penalty,
                "num_beams": 6,
                "max_length": max_summary_length,
                "min_length": min_summary_length,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
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

# Gradio function
def summarize_text(input_text, summary_length, verbosity):
    max_len = int(summary_length)
    length_pen = float(verbosity)
    summary = summarizer.summarize(
        input_text,
        max_summary_length=max_len,
        min_summary_length=max_len//2,
        length_penalty=length_pen
    )
    return summary, "Done!"  # Return summary and status

# Clear function
def clear_input():
    return "", "", "Ready"

# Custom CSS for a cool look
custom_css = """
body { font-family: 'Arial', sans-serif; background-color: #1e1e1e; color: #ffffff; }
.gradio-container { max-width: 900px; margin: 20px auto; padding: 20px; background: #2b2b2b; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
h1 { color: #00d4ff; text-align: center; font-size: 2.5em; margin-bottom: 20px; text-shadow: 0 0 10px rgba(0, 212, 255, 0.5); }
textarea { background-color: #333333 !important; color: #ffffff !important; border: 1px solid #00d4ff !important; border-radius: 8px; }
button { background: linear-gradient(45deg, #00d4ff, #007bff); border: none; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: all 0.3s; }
button:hover { transform: scale(1.05); box-shadow: 0 0 15px rgba(0, 212, 255, 0.7); }
.slider { background: #444444; border-radius: 10px; }
.label { color: #00d4ff; font-weight: bold; }
.output_text { background-color: #333333; color: #e0e0e0; border: 1px solid #00d4ff; border-radius: 8px; padding: 10px; }
.status { font-size: 0.9em; color: #00ff00; text-align: center; }
"""

# Gradio interface with Blocks
with gr.Blocks(title="Super Summarizer", css=custom_css) as interface:
    gr.Markdown("# Super Summarizer")
    gr.Markdown("Transform your text into concise, cool summaries with ease!")

    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(
                lines=10, placeholder="Drop your text here...", label="Input Text",
                elem_classes="input_text"
            )
            output_box = gr.Textbox(
                label="Your Summary", lines=5, interactive=False, elem_classes="output_text"
            )
        with gr.Column(scale=1):
            summary_length = gr.Slider(128, 512, step=64, value=256, label="Summary Length (tokens)")
            verbosity = gr.Slider(0.5, 2.0, step=0.1, value=1.0, label="Verbosity (Length Penalty)")
            with gr.Row():
                submit_btn = gr.Button("Summarize", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")
            status = gr.Textbox(value="Ready", label="Status", interactive=False, elem_classes="status")

    # Bind actions
    submit_btn.click(
        fn=summarize_text,
        inputs=[input_box, summary_length, verbosity],
        outputs=[output_box, status],
        _js="() => {document.querySelector('.status').value = 'Summarizing...';}"  # JS for progress
    )
    clear_btn.click(
        fn=clear_input,
        inputs=[],
        outputs=[input_box, output_box, status]
    )

# Launch
if __name__ == "__main__":
    interface.launch()
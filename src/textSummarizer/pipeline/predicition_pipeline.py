import logging
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_evaluation import ModelEvaluation
from transformers import AutoTokenizer, pipeline


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.tokenizer = None
        self.summarization_pipeline = None
        self.model_evaluation = None
        self._load_pipeline()
        self._setup_model_evaluation()

    def _load_pipeline(self):
        """Load the tokenizer and summarization pipeline."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            self.summarization_pipeline = pipeline(
                "summarization",
                model=self.config.model_path,
                tokenizer=self.tokenizer,
            )
        except Exception as e:
            logging.error(f"Error loading pipeline: {e}")
            raise

    def _setup_model_evaluation(self):
        """Initialize the model evaluation component."""
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            self.model_evaluation = ModelEvaluation(config=model_evaluation_config)
        except Exception as e:
            logging.error(f"Error setting up model evaluation: {e}")
            raise

    def predict(self, text):
        """Generate a summary for the given text."""
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Input text must be a non-empty string.")
            
            gen_kwargs = {
                "length_penalty": 0.8,
                "num_beams": 8,
                "max_length": 128,
                "min_length": 30  # Ensure min_length is set
            }

            logging.info("Received input text for summarization.")
            output = self.summarization_pipeline(text, **gen_kwargs)[0]["summary_text"]

            logging.info("Generated summary successfully.")
            return output

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

    def evaluate(self):
        """Evaluate the model using the evaluation component."""
        try:
            if not self.model_evaluation:
                raise RuntimeError("Model evaluation component is not initialized.")
            
            logging.info("Starting model evaluation.")
            self.model_evaluation.evaluate()
            logging.info("Model evaluation completed successfully.")

        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise

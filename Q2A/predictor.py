import torch
import pytorch_lightning as pl
from model import ModelModule
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer
)


class PythonPredictor:
    def __init__(self, config):
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        self.model = ModelModule.load_from_checkpoint(checkpoint_path="C:\Users\87547\Desktop\epoch=5-step=156.ckpt")

    def predict(self, payload):
        inputs = self.tokenizer.encode_plus(payload["text"], return_tensors="pt")
        predictions = self.model(**inputs)[0]
        if (predictions[0] > predictions[1]):
            return {"class": "unacceptable"}
        else:
            return {"class": "acceptable"}
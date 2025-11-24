import logging
from abc import ABC, abstractmethod

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)
    
class DPREmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, is_question: bool):

        if is_question:
            self.encoder = DPRQuestionEncoder.from_pretrained(model_name)
            self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        else:
            self.encoder = DPRContextEncoder.from_pretrained(model_name)
            self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
      
        self.is_question = is_question
        
    def create_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.encoder(**inputs).pooler_output.detach().numpy()
        return outputs

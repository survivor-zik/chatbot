# Load model directly
from llama_index.embeddings import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from modules.prompts import prompt
from llama_index.vector_stores import ChromaVectorStore
from llama_index import VectorStoreIndex, ServiceContext, load_index_from_storage
from llama_index.storage.storage_context import StorageContext
import torch
from llama_index.prompts.prompts import SimpleInputPrompt

class Chatbot:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("ericzzz/falcon-rw-1b-chat")
        model = AutoModelForCausalLM.from_pretrained("ericzzz/falcon-rw-1b-chat",device_map='auto',offload_folder='./model')
        self.embeddings = HuggingFaceEmbedding("WhereIsAI/UAE-Large-V1", device=self.device)
        self.prompt = SimpleInputPrompt(prompt)
        self.llm = HuggingFaceLLM(
            model, tokenizer, self.device ,system_prompt=self.prompt,offload_folder='./model'
        )
        

    def get_response(self, input_text):
        input_ids = self.tokenizer.encode(
            input_text + self.tokenizer.eos_token, return_tensors="pt"
        ).to(self.device)
        output_ids = self.model.generate(
            input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response


print(torch.cuda.is_available())

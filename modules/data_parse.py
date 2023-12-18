from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import download_loader
from fastapi import UploadFile

def load_pdf(file:UploadFile):
    PDFReader=download_loader("PDFReader")
    loader=PDFReader()
    documents=loader.load_data(file=file)

embed_model=HuggingFaceEmbedding("WhereIsAI/UAE-Large-V1",device="cuda")
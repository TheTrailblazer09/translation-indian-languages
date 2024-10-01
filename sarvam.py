# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from PyPDF2 import PdfReader


tokenizer = AutoTokenizer.from_pretrained("ibm/ia-multilingual-transliterated-roberta")
model = AutoModelForMaskedLM.from_pretrained("ibm/ia-multilingual-transliterated-roberta")

pdf_path='./khan101.pdf'
reader=PdfReader(pdf_path)
text=""

for page in reader.pages:
    text+=page.extract_text()

print(text)

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding='max_length')
outputs = model(**inputs)


# Decode the transliterated output (Note: this is an example. You may need post-processing based on model behavior)
transliterated_text = tokenizer.decode(torch.argmax(outputs.logits, dim=-1)[0])

print(transliterated_text)
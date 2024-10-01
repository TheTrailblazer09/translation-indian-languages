# Usage
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained('sarvamai/OpenHathi-7B-Hi-v0.1-Base')
model = LlamaForCausalLM.from_pretrained('sarvamai/OpenHathi-7B-Hi-v0.1-Base', torch_dtype=torch.bfloat16)

prompt = "मैं एक अच्छा हाथी हूँ"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
generated_text =tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(generated_text)

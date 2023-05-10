import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

# Import necessary libraries for the chatbot model
import torch
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM

# Initialize the chatbot model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
model = LlamaForCausalLM.from_pretrained(
    "chainyo/alpaca-lora-7b",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Set the chatbot model to evaluation mode
model.eval()

generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
)

def generate_prompt(instruction: str, input_ctxt: str = None) -> str:
    if input_ctxt:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_ctxt}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

# Constant scientific context
input_ctxt = "The answer must be scientific, factual and precise."


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        # Generate the prompt with the text and context
        prompt = generate_prompt(text, input_ctxt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        output.append(response)

    return SimpleText(dict(text=output))

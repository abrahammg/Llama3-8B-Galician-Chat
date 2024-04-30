---
license: apache-2.0
datasets:
- irlab-udc/alpaca_data_galician
language:
- gl
- en
---

# Galician Fine-Tuned LLM Model

This repository contains a large language model (LLM) adapter fine-tuned using the LLaMA Factory library. The base model used for fine-tuning was Meta's `LLaMA 3-8B`.

## Model Description

This LLM model has been specifically fine-tuned to understand and generate text in Galician. It was refined using a modified version of the [irlab-udc/alpaca_data_galician](https://huggingface.co/datasets/irlab-udc/alpaca_data_galician) dataset, enriched with synthetic data to enhance its text generation and comprehension capabilities in specific contexts.

### Technical Details

- **Base Model**: Meta's LLaMA 3 8B Instruct
- **Fine-Tuning Platform**: LLaMA Factory
- **Infrastructure**: Finisterrae III, CESGA
- **Dataset**: [irlab-udc/alpaca_data_galician](https://huggingface.co/datasets/irlab-udc/alpaca_data_galician) (with modifications)
- **Fine-Tuning Objective**: To improve text comprehension and generation in Galician.

### Trainning parameters

The project is still in the testing phase, and the training parameters will continue to vary to find the values that result in a more accurate model. Currently, the model is trained with a set of **5000 random entries** from the dataset and the following values:

- num_train_epochs=3.0
- finetuning_type="lora"
- per_device_train_batch_size=2
- gradient_accumulation_steps=4
- lr_scheduler_type="cosine"
- learning_rate=5e-5
- max_grad_norm=1.0

## Testing The Model

Below, we have several examples of question-answer pairs comparing the pure Llama3 model with the Llama3 model fine tunned for the Galician language.

*Llama3-8B*
```bash
User: Cal é a capital de Canadá?
Assistant: Não! Cal não é a capital de Canadá. A capital de Canadá é Ottawa.
```
*Llama3-8B-Galician-Chat*
```bash
User: Cal é a capital de Canadá?
Assistant: A capital de Canadá é Ottawa.
```
*Llama3-8B*
```bash
User: Cantos habitantes ten Galicia?
Assistant: Según el Instituto Nacional de Estadística (INE) de España, la población residente en Galicia al 1 de enero de 2021 era de aproximadamente 2.79 millones de personas.
```
*Llama3-8B-Galician-Chat*
```bash
User: Cantos habitantes ten Galicia?
Assistant: Segundo as últimas estimacións, Galicia ten uns 2,8 millóns de habitantes.
```

## How to Use the Model

To use this model, follow the example code provided below. Ensure you have the necessary libraries installed (e.g., Hugging Face's `transformers`).

### Installation

```bash
pip install transformers
pip install bitsandbytes
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install llmtuner
```

### Run the model

```bash
from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc

chat_model = ChatModel(dict(
  model_name_or_path="unsloth/llama-3-8b-Instruct-bnb-4bit", # use bnb-4bit-quantized Llama-3-8B-Instruct model
  adapter_name_or_path="model",            # load the saved LoRA adapters
  finetuning_type="lora",                  # same to the one in training
  template="llama3",                     # same to the one in training
  quantization_bit=4,                    # load 4-bit quantized model
  use_unsloth=True,                     # use UnslothAI's LoRA optimization for 2x faster generation
))

messages = []
while True:
  query = input("\nUser: ")
  if query.strip() == "exit":
    break

  if query.strip() == "clear":
    messages = []
    torch_gc()
    print("History has been removed.")
    continue

  messages.append({"role": "user", "content": query})     # add query to messages
  print("Assistant: ", end="", flush=True)
  response = ""
  for new_text in chat_model.stream_chat(messages):      # stream generation
    print(new_text, end="", flush=True)
    response += new_text
  print()
  messages.append({"role": "assistant", "content": response}) # add response to messages

torch_gc()
```
## Citation

```markdown
@misc{Llama3-8B-Galician-Chat,
  author = {Abraham Martínez Gracia},
  title = {Llama3-8B-Galician-Chat: A finetuned chat model for Galician language},
  year = {2024},
  url = {https://huggingface.co/abrahammg/Llama3-8B-Galician-Chat}
}
```

## Acknowledgement

[meta-llama/llama3](https://github.com/meta-llama/llama3)

[hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

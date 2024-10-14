
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

def tokenize_function(examples):
    # Combine 'about_me' and 'context' for the full context
    full_context = examples['about_me'] + ' ' + examples['context']
    
    # Create the prompt using the full context
    prompt = f"Question: {examples['question']}\nContext: {full_context}\nAnswer:"
    response = examples['response']
    
    # Tokenize inputs and labels
    tokenized_input = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized_output = tokenizer(response, truncation=True, padding="max_length", max_length=512)
    
    # Combine input and output (GPT-2 is autoregressive)
    input_ids = tokenized_input["input_ids"] + tokenized_output["input_ids"]
    
    # Create the labels (output sequence should be the entire concatenated sequence)
    labels = [-100] * len(tokenized_input["input_ids"]) + tokenized_output["input_ids"]
    
    return {
        "input_ids": input_ids, 
        "attention_mask": [1] * len(input_ids), 
        "labels": labels
    }


# Load model and tokenizer
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load datasets
train_dataset = load_dataset('json', data_files={"train": "/opt/ml/input/data/train/train_data.json"})['train']
test_dataset = load_dataset('json', data_files={"test": "/opt/ml/input/data/test/test_data.json"})['test']

tokenized_train = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names)
tokenized_test = test_dataset.map(tokenize_function, remove_columns=test_dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="s3://falcon-artifact/", # default value is "/opt/ml/model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_dir="/opt/ml/logs",
    logging_steps=100,
    fp16=True,
)

# Trainer setup
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

trainer.train()

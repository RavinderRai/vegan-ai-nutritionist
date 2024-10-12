
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

def tokenize_function(examples):
    full_context = examples['about_me'] + ' ' + examples['context']
    prompt = f"Question: {examples['question']}\nContext: {full_context}\nAnswer:"
    response = examples['response']
    
    tokenized_input = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized_output = tokenizer(response, truncation=True, padding="max_length", max_length=512)
    
    input_ids = tokenized_input["input_ids"] + tokenized_output["input_ids"][1:]  # Remove BOS token
    labels = [-100] * len(tokenized_input["input_ids"]) + tokenized_output["input_ids"][1:]
    
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

# Load model and tokenizer
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load datasets
train_dataset = load_dataset('json', data_files={"train": "/opt/ml/input/data/train/train_data.json"})['train']
test_dataset = load_dataset('json', data_files={"test": "/opt/ml/input/data/test/test_data.json"})['test']

tokenized_train = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names)
tokenized_test = test_dataset.map(tokenize_function, remove_columns=test_dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="/opt/ml/model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_dir="/opt/ml/logs",
    logging_steps=100,
    fp16=True,
)

# Trainer setup
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

trainer.train()

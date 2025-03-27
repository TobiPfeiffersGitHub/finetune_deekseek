import os
import torch
import yaml
import mlflow
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from huggingface_hub import login, create_repo, upload_folder
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType
from dotenv import load_dotenv

load_dotenv()

login(os.getenv("hf_key"))
wandb.login(key=os.getenv("wandb_key"))
run = wandb.init(
    project="fine_tune_deepseek_for_customersupport",
    job_type="training",
    anonymous="allow"
)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

sytem_prompt = """
You are a helpful customer support assistant. Read the inquiry and generate a professional response.

Customer: "{}"

Response:
<think>Understand the request and generate a courteous, helpful reply.
<response>"{}"
"""

# Load model & tokenizer
model_name = config["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,       # Force FP32 on CPU/MPS
    device_map={"": torch.device("mps" if torch.backends.mps.is_available() else "cpu")},
    low_cpu_mem_usage=True
)


# Use Apple MPS if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model = model.to(device)

# Prepare dataset
dataset = load_dataset("AabirDey/job-queries-and-customer-service")
EOS = tokenizer.eos_token

def format_prompt(example):
    inputs = example['instruction']
    outputs = example['output']
    return {"text": [sytem_promt.format(i, o) + EOS for i, o in zip(inputs, outputs)]}

dataset = dataset.map(format_prompt, batched=True)

# Tokenize
def tokenize(example):
    return (
        tokenizer(
            example["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=config["max_seq_length"]
        )
    )

tokenized = dataset.map(tokenize, batched=True)

# Finetuning with LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=config["output_dir"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    num_train_epochs=config["num_train_epochs"],
    learning_rate=0.0002,
    max_steps=config["max_steps"],
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    bf16=False,
    fp16=False,
    report_to=["wandb"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
2
trainer.train()
wandb.finish()


# Save model
create_repo(config["repo_name"], exist_ok=True)
upload_folder(
    repo_id=config["repo_name"],
    folder_path=f"{config['output_dir']}/checkpoint-{config['max_steps']}",
    path_in_repo=".",
    commit_message="Upload fine-tuned model"
)
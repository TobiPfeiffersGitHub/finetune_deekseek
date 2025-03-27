import os
import torch
import yaml
import mlflow
import wandb
from datasets import load_dataset
from transformers import TrainingArguments
from huggingface_hub import login, create_repo, upload_folder
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
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

sytem_promt = """
You are a helpful customer support assistant. Read the inquiry and generate a professional response.

Customer: "{}"

Response:
<think>Understand the request and generate a courteous, helpful reply.
<response>"{}"
"""

# Load model
model, tokenizer = (
    FastLanguageModel
    .from_pretrained(
        model_name = config["model_name"],
        max_seq_length = config["max_seq_length"],
        dtype = None,
        load_in_4bit = config["load_in_4bit"],
        token = os.getenv("hf_key")
    )
)

# Prepare dataset
dataset = load_dataset("AabirDey/job-queries-and-customer-service")
EOS = tokenizer.eos_token

def format_prompt(example):
    inputs = example['instruction']
    outputs = example['output']
    return {"text": [sytem_promt.format(i, o) + EOS for i, o in zip(inputs, outputs)]}

dataset = dataset.map(format_prompt, batched=True)

# Finetuning with LoRA
model_lora = (
    FastLanguageModel
    .get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = config["seed"],
        use_rslora = False,
        loftq_config = None
    )
)

trainer = SFTTrainer(
    model = model_lora,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    dataset_text_field="text",
    max_seq_length = config['max_seq_length'],
    dataset_num_proc = 2,

    args = TrainingArguments(
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs = config["num_train_epochs"],
        warmup_steps = 5,
        max_steps = config["max_steps"],
        learning_rate = config['learning_rate'],
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed=config["seed"],
        output_dir=config["output_dir"]
    )
)

# Train & inference
trainer_stats = trainer.train()
wandb.log({"final_loss": trainer_stats.training_loss})
wandb.finish()

# Save model
create_repo(config["repo_name"], exist_ok=True)
upload_folder(
    repo_id=config["repo_name"],
    folder_path=f"{config['output_dir']}/checkpoint-{config['max_steps']}",
    path_in_repo=".",
    commit_message="Upload fine-tuned model"
)
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
from peft import get_peft_model, LoraConfig
from datasets import load_dataset

model_name = "google/gemma-3-12b-pt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             device_map="auto", 
                                             torch_dtype="auto")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],  
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

raw = load_dataset("text", data_files={"train": "trainset/labels_detok.txt"})
def tokenize(x):
    return tokenizer(x["text"], truncation=True, max_length=512)
ds = raw.map(tokenize, batched=True, remove_columns=["text"])
ds.set_format("torch", columns=["input_ids", "attention_mask"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="gemma-domain-adapt",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    fp16=True,
    num_train_epochs=3,
    logging_steps=100,
    save_total_limit=2,
    save_steps=500,
    weight_decay=0.01,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("gemma-domain-adapt")

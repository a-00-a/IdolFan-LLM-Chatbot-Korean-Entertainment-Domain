from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
import torch

def fine_tune(model, dataset, output_dir="./idolfan_lora", epochs=3):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=5e-4,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

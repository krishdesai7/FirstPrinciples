from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer 
from problem_solver import TPBenchData, OUTPUT_DIR

def main():
    dataset = load_dataset("ZhiqiGao/TPBench")
    dataset = dataset.train_test_split(test_size=0.1)
    model, tokenizer = TPBenchData.setup_model_and_tokenizer()
    processor = TPBenchData(tokenizer)
    train_dataset = processor.preprocess(dataset["train"])
    eval_dataset = processor.preprocess(dataset["test"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args = TPBenchData.create_training_arguments(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()

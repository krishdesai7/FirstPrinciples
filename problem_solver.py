import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from typing import Dict

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

MODEL_ID = "EleutherAI/llemma_7b"
OUTPUT_DIR = "llemma_7b_tpbench_finetune"

#QLoRA Config
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", # Attention layers
        "gate_proj", "down_proj", "up_proj", #FFN layers

    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
)

#4 bit quantization
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

class TPBenchData:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.physics_template = """### Physics Problem
{problem}

### Solution Approach
{solution}

### Final Answer
{answer}"""

    def format_problem_solution(self, example: Dict) -> str:
        """Format TPBench problem with reasoning structure"""
        problem = example.get("problem", "")
        solution = example.get("solution", "")
        answer = example.get("answer", "")

        formatted_problem = self.physics_template.format(
            problem=problem,
            solution=solution,
            answer=answer,
        )
        return formatted_problem
    
    def preprocess(self, dataset):
        """Preprocess TPBench dataset for training"""
        def tokenize_function(examples):
            #Format each example and tokenize
            formatted_text = [
                self.format_problem_solution(example)
                for example in examples
            ]

            model_inputs = self.tokenizer(formatted_text,
                                          max_length=MAX_SEQ_LENGTH, 
                                          truncation=True,
                                          padding="max_length"
                                          )
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        tokenized_dataset = dataset.map(tokenize_function,
                                        remove_columns=dataset.column_names,
                                        batched=True,
                                        )
        return tokenized_dataset

    def setup_model_and_tokenizer():
        """Initialize Llemma with QLoRA config"""

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=QUANTIZATION_CONFIG,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()
        return model, tokenizer
    
    def create_training_arguments():
        '''Create training arguments for finetuning'''
        return TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            logging_steps=10,
            learning_rate=LEARNING_RATE,
            fp16=True if DEVICE != "mps" else False,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps = 100,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="tensorboard",
            push_to_hub=False,
            gradient_checkpointing=True,
            optim="adamw_torch",
            adam_beta2=0.999,
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
        )
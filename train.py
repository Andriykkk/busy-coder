import argparse
import torch
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
import os
import copy
import json

IGNORE_INDEX = -100

# --- Utility Functions (from qlora.py) ---

def find_all_linear_names(model, bits):
    """Finds all linear layer names in the model for PEFT configuration."""
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_model_size_analysis(model):
    """
    Prints the number of trainable parameters, total parameters,
    percentage of trainable parameters, and the average bit size of model parameters.
    """
    trainable_params = 0
    all_param = 0
    total_bits = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        all_param += numel
        if param.requires_grad:
            trainable_params += numel

        # Get dtype size in bits
        dtype_size_bits = param.element_size() * 8
        total_bits += numel * dtype_size_bits

    avg_bits_per_param = total_bits / all_param if all_param > 0 else 0

    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}%"
    )
    print(f"average bits per parameter: {avg_bits_per_param:.2f}")

# --- Data Collator (from qlora.py, adapted for our purpose) ---

class DataCollatorForCausalLM(object):
    def __init__(self, tokenizer, train_on_source=False):
        self.tokenizer = tokenizer
        self.train_on_source = train_on_source

    def __call__(self, instances):
        # This collator is designed to work with the processed dataset,
        # which has 'source' and 'target' fields.
        sources = [instance['source'] for instance in instances]
        targets = [instance['target'] for instance in instances]

        # Tokenize
        tokenized_sources = self.tokenizer(sources, add_special_tokens=False)
        tokenized_targets = self.tokenizer(targets, add_special_tokens=False)

        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources['input_ids'],
            tokenized_targets['input_ids']
        ):
            source_ids = self.tokenizer.build_inputs_with_special_tokens(tokenized_source)
            target_ids = tokenized_target + [self.tokenizer.eos_token_id]

            input_ids.append(torch.tensor(source_ids + target_ids))
            if not self.train_on_source:
                labels.append(
                    torch.tensor([IGNORE_INDEX] * len(source_ids) + copy.deepcopy(target_ids))
                )
            else:
                labels.append(torch.tensor(copy.deepcopy(source_ids + target_ids)))

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'labels': labels,
        }
        return data_dict

# --- Data Preparation ---

def format_codeact_example(batch, tokenizer, source_max_len, target_max_len):
    """
    Formats a batch of examples from the xingyaoww/code-act dataset and
    filters them by length in a single, efficient pass.
    """
    all_sources = []
    all_targets = []
    
    role_map = {
        "system": "### System: ",
        "user": "### User: ",
        "assistant": "### Assistant: "
    }
    
    for conversation in batch['conversations']:
        conversation_history = ""
        for turn in conversation:
            role = turn['role'].lower()
            content = turn['content']
            prefix = role_map.get(role, f"### {role.capitalize()}: ")
            
            if role == "assistant":
                source = conversation_history + prefix
                target = content

                # Tokenize and check length right here to be efficient
                source_tokens = tokenizer(source, add_special_tokens=False)['input_ids']
                target_tokens = tokenizer(target, add_special_tokens=False)['input_ids']

                if len(source_tokens) <= source_max_len and len(target_tokens) <= target_max_len:
                    all_sources.append(source)
                    all_targets.append(target)

            conversation_history += prefix + content + "\n"
            
    return {"source": all_sources, "target": all_targets}


def prepare_data(tokenizer, dataset_name, train_split, eval_split, dataset_path, source_max_len, target_max_len, num_proc):
    """
    Loads, processes, and saves the dataset splits using multiple processes.
    If the processing arguments change, it automatically re-processes the data.
    """
    args_path = os.path.join(dataset_path, "data_args.json")
    
    # Check if a cached dataset exists and if the arguments match
    if os.path.exists(dataset_path) and os.path.exists(args_path):
        print(f"Found cached dataset at {dataset_path}.")
        with open(args_path, "r") as f:
            saved_args = json.load(f)

        args_match = (
            saved_args.get("dataset_name") == dataset_name and
            saved_args.get("train_split") == train_split and
            saved_args.get("eval_split") == eval_split and
            saved_args.get("source_max_len") == source_max_len and
            saved_args.get("target_max_len") == target_max_len
        )

        if args_match:
            print("Dataset arguments match. Loading and validating cached data...")
            processed_datasets = load_dataset(dataset_path)

            # Validate the structure of the cached dataset
            required_columns = {'source', 'target'}
            if 'train' in processed_datasets and required_columns.issubset(processed_datasets['train'].column_names):
                print("Cached data is valid. Skipping re-processing.")
                return processed_datasets
            else:
                print("Cached data is invalid or missing required columns ('source', 'target'). Re-processing data.")
        else:
            print("Dataset arguments have changed. Re-processing data.")
    
    print(f"Processing data from source: {dataset_name}...")
    raw_datasets = load_dataset(dataset_name)
    
    from functools import partial
    format_and_filter = partial(
        format_codeact_example,
        tokenizer=tokenizer,
        source_max_len=source_max_len,
        target_max_len=target_max_len
    )

    def process_split(split_name):
        raw_split = raw_datasets[split_name]
        
        # --- Count total potential samples (fast pass) ---
        total_potential_samples = sum(1 for conv in raw_split['conversations'] for turn in conv if turn['role'].lower() == 'assistant')
        
        # --- Format, filter, and tokenize (main pass) ---
        processed_split = raw_split.map(
            format_and_filter,
            batched=True,
            remove_columns=["id", "conversations"],
            num_proc=num_proc
        )
        
        final_samples = len(processed_split)
        truncated_samples = total_potential_samples - final_samples
        
        print(f"\n--- {split_name} Split Processing ---")
        print(f"Total potential samples: {total_potential_samples}")
        print(f"Samples after filtering: {final_samples}")
        if total_potential_samples > 0:
            print(f"Samples truncated: {truncated_samples} ({truncated_samples / total_potential_samples:.2%})")
        print("------------------------------------")

        return processed_split

    train_dataset = process_split(train_split)
    eval_dataset = process_split(eval_split)

    processed_splits = DatasetDict({
        'train': train_dataset,
        'test': eval_dataset
    })

    print(f"\nSaving processed dataset splits to {dataset_path}...")
    os.makedirs(dataset_path, exist_ok=True)
    processed_splits.save_to_disk(dataset_path)
    
    current_args = {
        "dataset_name": dataset_name,
        "train_split": train_split,
        "eval_split": eval_split,
        "source_max_len": source_max_len,
        "target_max_len": target_max_len
    }
    with open(args_path, "w") as f:
        json.dump(current_args, f, indent=4)

    return processed_splits



def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning script for code-act style training.")

    # --- Model and Tokenizer ---
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Hugging Face model name or path.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for checkpoints.")
    
    # --- Dataset Arguments ---
    parser.add_argument("--dataset_name", type=str, default="xingyaoww/code-act", help="Source dataset name from Hugging Face Hub.")
    parser.add_argument("--dataset_path", type=str, default="./processed_data/code-act", help="Path to save/load the processed dataset.")
    parser.add_argument("--train_split", type=str, default="codeact", help="The split to use for training.")
    parser.add_argument("--eval_split", type=str, default="general", help="The split to use for evaluation.")
    parser.add_argument("--source_max_len", type=int, default=2048, help="Maximum source sequence length.")
    parser.add_argument("--target_max_len", type=int, default=512, help="Maximum target sequence length.")

    # --- Quantization Arguments ---
    parser.add_argument("--bits", type=int, default=4, help="Bits for quantization (4 or 8).")
    parser.add_argument("--quant_type", type=str, default="nf4", help="Quantization type (e.g., 'nf4', 'fp4').")
    parser.add_argument("--double_quant", action='store_true', default=True, help="Use double quantization.")

    # --- LoRA Arguments ---
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r dimension.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument("--lora_modules", type=str, default="all", help="Which modules to apply LoRA to.")

    # --- Training Hyperparameters ---
    parser.add_argument("--bf16", action='store_true', default=True, help="Use bfloat16 for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Learning rate scheduler type.")
    parser.add_argument("--gradient_checkpointing", action='store_true', default=True, help="Enable gradient checkpointing.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Training batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--max_steps", type=int, default=1875, help="Total number of training steps.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Max gradient norm for clipping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes to use for data processing.")

    # --- Saving Arguments ---
    parser.add_argument("--save_merged_model", action='store_true', help="Merge adapter and save full model.")
    parser.add_argument("--save_separated_model", action='store_true', help="Save adapter and save full model separately.(In development)")
    parser.add_argument("--save_tokenizer", action='store_true', help="Save tokenizer with the model.")

    args = parser.parse_args()

    print("--- Parsed Arguments ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("------------------------\n")

    set_seed(args.seed)

    # --- Load Tokenizer ---
    print(f"Loading tokenizer for model: {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Tokenizer loaded successfully.\n")

    # --- Load Model with Quantization ---
    print(f"Loading base model: {args.model_name_or_path} with {args.bits}-bit quantization...")
    
    # Check for bfloat16 support
    if args.bf16 and not torch.cuda.is_bf16_supported():
        print("Warning: bfloat16 is not supported on this GPU. Falling back to float16.")
        args.bf16 = False

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    print(f"Using compute dtype: {compute_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Base model loaded successfully.\n")

    # --- PEFT and LoRA Setup ---
    print("Setting up model for K-bit training and applying LoRA...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.lora_modules == "all":
        target_modules = find_all_linear_names(model, args.bits)
        print(f"Found all linear modules for LoRA: {target_modules}")
    else:
        target_modules = args.lora_modules.split(',')
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("LoRA configuration applied.\n")
    print_model_size_analysis(model)

    # --- Prepare Dataset ---
    processed_datasets = prepare_data(
        tokenizer, args.dataset_name, args.train_split, args.eval_split, args.dataset_path, args.source_max_len, args.target_max_len, args.num_proc
    )
    print("Dataset length:", len(processed_datasets['train']))

    # --- Setup Trainer ---
    print("\n--- Initializing Trainer ---")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps, # Evaluate at the same frequency as saving
        bf16=args.bf16,
        tf32=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 and args.bf16,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        remove_unused_columns=False,
        predict_with_generate=True,
    )
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)
    print("Available dataset splits:", processed_datasets.keys())
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=processed_datasets['train'],
        eval_dataset=processed_datasets['test'],
        data_collator=data_collator,
    )

    print("--- Starting Training ---")
    trainer.train()
    print("--- Training Finished ---")

    # Save the final adapter
    final_adapter_path = os.path.join(args.output_dir, "final_adapter")
    print(f"Saving final adapter to {final_adapter_path}")
    model.save_pretrained(final_adapter_path)

    # --- Save Merged Model and Tokenizer ---
    if args.save_merged_model:
        print("\n--- Merging and Saving Final Model ---")
        
        # Merge the LoRA adapter with the base model
        try:
            # merge_and_unload() unloads the LoRA weights and merges them with the base model
            model = model.merge_and_unload()
            print("Model merged successfully.")
        except Exception as e:
            print(f"Error during model merging: {e}")

        # Define the path for the final merged model
        final_model_path = os.path.join(args.output_dir, "final_merged_model")
        os.makedirs(final_model_path, exist_ok=True)

        print(f"Saving merged model to {final_model_path}")
        try:
            model.save_pretrained(final_model_path)
            print("Merged model saved successfully.")
        except Exception as e:
            print(f"Error saving merged model: {e}")

        if args.save_tokenizer:
            print(f"Saving tokenizer to {final_model_path}")
            try:
                tokenizer.save_pretrained(final_model_path)
                print("Tokenizer saved successfully.")
            except Exception as e:
                print(f"Error saving tokenizer: {e}")


if __name__ == "__main__":
    main()
    
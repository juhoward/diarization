from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import argparse

# login()

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_id,
                                                quantization_config=bnb_config,
                                                device_map="auto",
                                                use_flash_attention_2=True,
                                                trust_remote_code=True,
                                                resume_download=True)

    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_id,
                                            trust_remote_code=True,
                                            resume_download=True
                                            )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f'Saving model: /data/chat_models/foundation/{args.model_id}')
    model.save_pretrained('/data/chat_models/foundation/' + args.model_id)
    model.mode.save_config('/data/chat_models/foundation/' + args.model_id)
    tokenizer.save_pretrained('/data/chat_models/foundation/' + args.model_id)
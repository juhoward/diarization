from vision_dataset import VisionDatasetCreator, VisionDataset
import os
# os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:50'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

print(f'Visible devices:\n{[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}')
print(f'device peroperties:\n{[torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count())]}')
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizerFast

# torch.cuda.set_device(0)

phoroptermodel = False
# avg percentages of exam phase lengths
gs_len=0.14
sp_len=0.25
ac_len=0.50
cv_len=0.11
# percentage of samples taken from each exam phase
gs_ratio=0.15
sp_ratio=0.30
ac_ratio=0.40
cv_ratio=0.15
if phoroptermodel:
    # minumum dialogue lengths
    gs_min = 2
    sp_min = 4
    ac_min = 4
    cv_min = 4
    # maximum dialogue lengths
    gs_max = 6
    sp_max = 6
    ac_max = 6
    cv_max = 6
else:
    # minumum dialogue lengths
    gs_min = 2
    sp_min = 4
    ac_min = 6
    cv_min = 6
    # maximum dialogue lengths
    gs_max = 5
    sp_max = 10 # 14
    ac_max = 10 # 18
    cv_max = 10


sampling_strategy = dict(
    gs=[gs_len, gs_ratio, gs_min, gs_max],
    sp=[sp_len, sp_ratio, sp_min, sp_max],
    ac=[ac_len, ac_ratio, ac_min, ac_max],
    cv=[cv_len, cv_ratio, cv_min, cv_max]
)

if phoroptermodel:
    data_dir = '/data/datasets/phoropter_v3/'
    dataset_creator = VisionDatasetCreator(sampling_strategy, seed=42, assistant=False)
else:
    data_dir = '/data/datasets/Exam_v3/'
    # set seed to get randomization with reproducible results
    dataset_creator = VisionDatasetCreator(sampling_strategy, seed=42)

samples = 15
# 25 samples from each file in the training set, which has 21 files total
size = (32*samples)
dataset_creator.load(data_dir, 'train', size)
# 25 samples from each validation file, 3 files total
size = (5*samples)
dataset_creator.load(data_dir, 'val', size)
# 25 samples from each test file, 6 files total
# size = (8*samples)
# dataset_creator.load(data_dir, 'test', size)

train = VisionDataset(dataset_creator.dataset["train"])
val = VisionDataset(dataset_creator.dataset["val"])

def format_instruction(data):
    instruction = "You are an Optician conducting a vision exam. Use the dialogue below to create the Assistant's response that best guides Patient__ through a vision exam."
    dialogue_string = ''
    for i in data['dialogue']:
        dialogue_string += f'''{i['role']}: {i['content']}\n'''
    response_string = '' 
    for i in data['response']:
        response_string += f'''{i['role']}: {i['content']}\n'''
    return f'''###Instruction:\n{instruction}\n###Input:\n{dialogue_string}\n###Response:\n{response_string}'''

def format_mixtral(data, model_input=True):
    instruction = 'You are an Optician conducting a vision exam. Use the dialogue below to create an Assistant response that guides the Patient__ through a vision exam.'
    dialogue_string = ''
    for i in data['dialogue']:
        dialogue_string += f'''{i['role']}: {i['content']}\n'''
    response_string = ''
    for i in data['response']:
        response_string += f'''{i['role']}: {i['content']}\n'''
    if model_input:
        return f'''<s>[INST]{instruction}\n{dialogue_string}[/INST]{response_string}</s>'''
    else:
        return f'''<s>[INST]{instruction}\n{dialogue_string}[/INST]'''

max_seq_length = 0
for i in [train, val]:
    value = i.get_max_len(format_instruction)
    if value > max_seq_length:
        max_seq_length = value
    print(f'dataset length: {i.__len__()}  max sequence length: {i.get_max_len(format_instruction)}')
global_max_seq_length = 2048
if max_seq_length > global_max_seq_length:
    max_seq_length = global_max_seq_length
    print(f'WARNING: dataset max sequence length exceeds {global_max_seq_length}. Some inputs may be truncated.')
print(f'max_seq_length set to {max_seq_length}')


epochs = 20
mixtral=False
llama = False
phi2 = True
minicpm = False
if minicpm:
    from device_map import device_map
stopping = 'early-stop'

data_nm= data_dir.split('/')[-2]
train_len = str(train.__len__())
# model_id = "NousResearch/Llama-2-13b-chat-hf"
# cache_dir = '/data/chat-models/foundation/NousResearch/Llama-2-13b-chat-hf'

# model_id = "openchat/openchat-3.5-1210"
# cache_dir = '/data/chat-models/foundation/openchat/openchat-3.5-1210'

# model_id = "openchat/openchat-3.5-0106"
# cache_dir = '/data/chat-models/foundation/openchat/openchat-3.5-0106'

# model_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
# cache_dir = '/data/chat-models/foundation/mistralai/Mixtral-8x7B-Instruct-v0.1'

# model_id = "microsoft/phi-2"
# cache_dir = '/data/chat-models/foundation/microsoft/phi-2'

model_id = "google/gemma-2b-it"
cache_dir = '/data/chat-models/foundation/google/gemma-2b-it'

# model_id = 'intel/neural-chat-7b-v3-3'
# cache_dir = '/data/chat-models/foundation/intel/neural-chat-7b-v3-3'


model_name = model_id.split('/')[-1] + f"-int4-{data_nm}-{epochs}_{train_len}-{stopping}"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

if llama == True:
    model = LlamaForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             device_map="auto",
                                             use_flash_attention_2=True,
                                             do_sample=True,
                                             use_cache=False,
                                             cache_dir= cache_dir)

    model.config.pretraining_tp = 1

    tokenizer = LlamaTokenizerFast.from_pretrained(model_id,
                                                   cache_dir= cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
if mixtral == True:
    from transformers import MixtralForCausalLM
    model = MixtralForCausalLM.from_pretrained(model_id,
                                            quantization_config=bnb_config,
                                            device_map="auto",
                                            attn_implementation="flash_attention_2",
                                            do_sample=True,
                                            cache_dir= cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              add_eos_token=True,
                                              cache_dir= cache_dir)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id =  tokenizer.unk_token_id
    tokenizer.padding_side = "right"
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
if phi2 == True:
    print(f'Loading PhiForCausalLM: {model_id}')
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                quantization_config=bnb_config,
                                                device_map="auto",
                                                # microsoft is having issues with FA-2
                                                # attn_implementation="flash_attention_2",
                                                torch_dtype=torch.bfloat16,
                                                temperature=.9,
                                                do_sample=True,
                                                cache_dir= cache_dir,
                                                trust_remote_code=True)

    print(f'AutoTokenizer: {model_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              cache_dir= cache_dir,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
        # "dense",
        # "fc1",
        # "fc2"
    ]
else:
    print(f'Loading AutoModelForCausalLM: {model_id}')
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                quantization_config=bnb_config,
                                                device_map=device_map if minicpm else "auto",
                                                # attn_implementation="flash_attention_2", # use_flash_attention_2=True,
                                                temperature=.9,
                                                do_sample=True,
                                                cache_dir= cache_dir,
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True)

    print(f'{model.hf_device_map}')
    # model.config.pretraining_tp = 1
    print(f'AutoTokenizer: {model_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              cache_dir= cache_dir,
                                              torch_dtype=torch.bfloat16,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# specify the linear layers of the mistral-7b and openchat models per the PEFT paper

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        inference_mode=False,
        lora_alpha=32 if phi2 else 16,
        lora_dropout=0.1,
        r=32 if phi2 else 64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= target_modules if not llama else None,
        modules_to_save = ["lm_head","embed_tokens"] if phi2 else None
)

# prepare model for training
print(f'Applying PEFT and LoRA parameters...\n{peft_config}')
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# target_modules.append("mlp")
# max_memory = get_balanced_memory(
#     model,
#     max_memory=None,
#     no_split_module_classes=target_modules,
#     dtype='bfloat16',
#     low_zero=False,
# )

# device_map = infer_auto_device_map(
#     model,
#     max_memory=max_memory,
#     no_split_module_classes=target_modules,
#     dtype='bfloat16'
# )

# model = dispatch_model(model, device_map=device_map)





from transformers import EarlyStoppingCallback

# Early stopping patience (number of epochs without improvement)
early_stopping_patience = 2

# Early stopping threshold (minimum relative improvement to continue training)
early_stopping_threshold = -0.001

# Create the callback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience, early_stopping_threshold)

from transformers import TrainingArguments


args = TrainingArguments(
    output_dir=model_name, 
    num_train_epochs=epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",# "paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy='epoch',
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    disable_tqdm=True, # disable tqdm since with packing values are incorrect
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss'
)

from trl import SFTTrainer
print(f'SFTTrainer: {model_id}')
trainer = SFTTrainer(
    model=model,
    train_dataset=train.data,
    eval_dataset=val.data,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_mixtral if mixtral else format_instruction,
    args=args,
    callbacks=[early_stopping_callback]
)

# train
trainer.train() # there will not be a progress bar since tqdm is disabled

print("Saving model:   ", model_name)
# save model
trainer.save_model()
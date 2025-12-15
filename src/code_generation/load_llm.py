from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

def load_llm(max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True ):
    # load_in_4bit - Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-3B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    FastLanguageModel.for_inference(model)
    return tokenizer, model


def get_llm_answer(asks, tokenizer, model,
                   max_new_tokens=500, use_cache=False,
                   temperature=1.5, min_p=0.1
                   ):
    inputs = tokenizer.apply_chat_template(
        asks,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = max_new_tokens, use_cache = use_cache,
                             temperature = temperature, min_p = min_p)
    return tokenizer.batch_decode(outputs)

def generate_code_from_brain(words, tokenizer, model):
    basic_prompt = "Write a python function for the task: "
    return get_llm_answer(basic_prompt + " ".join(words), tokenizer, model)

from openai import OpenAI
from combine_generations_original_data import load_jsonl
import json
import os
import time

client = OpenAI(
    api_key=''
)

def create_batch(file):
    batch_input_file_id = file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "causal chains generation job"
        }
    )

def upload_file(jsonl_file: str):
    batch_input_file = client.files.create(
        file=open(jsonl_file, "rb"),
        purpose="batch"
    )
    return batch_input_file

def upload_to_openai(file_to_be_uploaded):
    batch_file = upload_file(jsonl_file=file_to_be_uploaded)
    create_batch(batch_file)
    print("File Uploaded Successfully")


def transform_completion_to_dict(chat_completion, sample):
    return {
        "custom_id": sample['custom_id'],
        "id": chat_completion.id,
        "choices": [
            {
                "finish_reason": choice.finish_reason,
                "index": choice.index,
                "logprobs": choice.logprobs,
                "message": {
                    "content": choice.message.content,
                    "refusal": choice.message.refusal,
                    "role": choice.message.role,
                    "audio": choice.message.audio,
                    "function_call": choice.message.function_call,
                    "tool_calls": choice.message.tool_calls
                },
                "stop_reason": choice.stop_reason
            } for choice in chat_completion.choices
        ],
        "created": chat_completion.created,
        "model": chat_completion.model,
        "object": chat_completion.object,
        "service_tier": chat_completion.service_tier,
        "system_fingerprint": chat_completion.system_fingerprint,
        "usage": {
            "completion_tokens": chat_completion.usage.completion_tokens,
            "prompt_tokens": chat_completion.usage.prompt_tokens,
            "total_tokens": chat_completion.usage.total_tokens,
            "completion_tokens_details": chat_completion.usage.completion_tokens_details,
            "prompt_tokens_details": chat_completion.usage.prompt_tokens_details
        }
    }

def transform_completion_to_dict_deepseek(chat_completion, sample):
    return {
        "custom_id": sample['custom_id'],
        "id": chat_completion.id,
        "choices": [
            {
                "finish_reason": choice.finish_reason,
                "index": choice.index,
                "logprobs": choice.logprobs,
                "message": {
                    "content": choice.message.content,
                    "refusal": choice.message.refusal,
                    "role": choice.message.role,
                    "audio": choice.message.audio,
                    "function_call": choice.message.function_call,
                    "tool_calls": choice.message.tool_calls
                },
                "matched_stop": choice.matched_stop
            } for choice in chat_completion.choices
        ],
        "created": chat_completion.created,
        "model": chat_completion.model,
        "object": chat_completion.object,
        "service_tier": chat_completion.service_tier,
        "system_fingerprint": chat_completion.system_fingerprint,
        "usage": {
            "completion_tokens": chat_completion.usage.completion_tokens,
            "prompt_tokens": chat_completion.usage.prompt_tokens,
            "total_tokens": chat_completion.usage.total_tokens,
            "completion_tokens_details": chat_completion.usage.completion_tokens_details,
            "prompt_tokens_details": chat_completion.usage.prompt_tokens_details
        }
    }

def generate_with_nemotron(file_with_prompts, output_file):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=""
    )
    with open(output_file, 'a') as file:
        for sample in load_jsonl(file_with_prompts):
            completion = client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
                messages=sample['body']['messages'],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                seed=256
            )
            nemotron_dict = transform_completion_to_dict(completion, sample)

            file.write(json.dumps(nemotron_dict) + "\n")

def generate_with_deepseek(file_with_prompts, output_file):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=""
    )
    with open(output_file, 'a') as file:
        for sample in load_jsonl(file_with_prompts):
            message = [{"role": "system",
                         "content": "<|im_start|>user\n" + sample['body']['messages'][0]['content'] + "<|im_end|>\n<|im_start|>assistant\n"}]
            completion = client.chat.completions.create(
                model="deepseek-ai/deepseek-r1",
                messages=message,
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                seed=256
            )
            deepseek_dict = transform_completion_to_dict_deepseek(completion, sample)

            file.write(json.dumps(deepseek_dict) + "\n")

            time.sleep(5)

def generate_with_qwq(file_with_prompts, output_file):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=""
    )
    with open(output_file, 'a') as file:
        for sample in load_jsonl(file_with_prompts):
            completion = client.chat.completions.create(
                model="qwen/qwq-32b",
                messages=sample['body']['messages'],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                seed=256
            )
            qwq_dict = transform_completion_to_dict(completion, sample)

            file.write(json.dumps(qwq_dict) + "\n")

def generate_with_mixtral(file_with_prompts, output_file):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=""
    )
    with open(output_file, 'a') as file:
        for sample in load_jsonl(file_with_prompts):
            completion = client.chat.completions.create(
                model="mistralai/mixtral-8x22b-instruct-v0.1",
                messages=sample['body']['messages'],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                seed=256
            )
            mixtral_dict = transform_completion_to_dict(completion, sample)

            file.write(json.dumps(mixtral_dict) + "\n")

def generate_with_phi(file_with_prompts, output_file):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=""
    )
    with open(output_file, 'a') as file:
        for sample in load_jsonl(file_with_prompts)[483:]:
            completion = client.chat.completions.create(
                model="microsoft/phi-4-mini-instruct",
                messages=sample['body']['messages'],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,  # 4096
                frequency_penalty=0,
                presence_penalty=0,
                seed=256
            )
            f_dict = transform_completion_to_dict(completion, sample)

            file.write(json.dumps(f_dict) + "\n")

def generate_with_mistral_nemo(file_with_prompts, output_file):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=""
    )

    with open(output_file, 'a') as file:
        for sample in load_jsonl(file_with_prompts):
            completion = client.chat.completions.create(
                model="nv-mistralai/mistral-nemo-12b-instruct",
                messages=sample['body']['messages'],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                seed=256
            )
            f_dict = transform_completion_to_dict(completion, sample)

            file.write(json.dumps(f_dict) + "\n")

def generate_with_llama_3(file_with_prompts, output_file):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=""
    )
    with open(output_file, 'a') as file:
        for sample in load_jsonl(file_with_prompts):
            completion = client.chat.completions.create(
                model="meta/llama3-70b-instruct",
                messages=sample['body']['messages'],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                seed=256
            )
            f_dict = transform_completion_to_dict(completion, sample)

            file.write(json.dumps(f_dict) + "\n")

def upload_to_lm(model_name, file_with_prompts, output_file):
    directory = os.path.dirname(output_file)
    # Create directories if they do not exist
    os.makedirs(directory, exist_ok=True)
    if model_name == "deepseek_r1":
        generate_with_deepseek(file_with_prompts, output_file)
    elif model_name == "llama31_nemotron":
        generate_with_nemotron(file_with_prompts, output_file)
    elif model_name == "llama_3_70b":
        generate_with_llama_3(file_with_prompts, output_file)
    elif model_name == "mistral_nemo":
        generate_with_mistral_nemo(file_with_prompts, output_file)
    elif model_name == "mixtral":
        generate_with_mixtral(file_with_prompts, output_file)
    elif model_name == "phi_4-mini":
        generate_with_phi(file_with_prompts, output_file)
    elif model_name == "qwq_32b":
        generate_with_qwq(file_with_prompts, output_file)
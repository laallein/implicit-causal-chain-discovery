import pandas as pd
import json

from combine_generations_original_data import load_jsonl
from analyze_chains import exp1_generate_prompt,exp2_generate_prompt, exp1_passive_generate_prompt, exp2_passive_generate_prompt
import os

def generate_prompt(cause, effect):
    return f"""A causal chain is a sequence of events in which each event directly causes the next, forming a connected series of cause-and-effect relationships. Unfolding a causal chain means identifying and linking individual events. A step of the chain presents only one noun phrase containing the event. Unfold all possible causal chains that connect {cause} (initial cause) to {effect} (final effect) and separate the steps of the chain with the token <step>, and the chains with the token <chain>."""

def add_prompt(cause, effect):
    return generate_prompt(cause, effect)

def process_input_file(file_name: str,
                       cause_col: str,
                       effect_col: str,
                       task = "generate",
                       experiment_num = 1,
                       put_in_passive = False):
    # First load file
    if ".xlsx" in file_name:
        df = pd.read_excel(file_name, sheet_name="Mastertable")
    elif ".jsonl" in file_name:
        df = pd.DataFrame(load_jsonl(file_name))
    elif ".csv" in file_name:
        df = pd.read_csv(file_name)

    # Add custom ids if not in df
    if 'custom_id' not in df.columns:
        df = create_custom_ids(df, file_name)

    # Add keys needed for Batch upload to OpenAI
    if task == "generate":
        df['prompt'] = df.apply(lambda row: add_prompt(row[cause_col], row[effect_col]), axis=1)
    elif task == "analyze":
        if experiment_num == 1:
            if not put_in_passive:
                df['prompt'] = df.apply(lambda row: exp1_generate_prompt(row[cause_col], row[effect_col]), axis=1)
            else:
                df['prompt'] = df.apply(lambda row: exp1_passive_generate_prompt(row[cause_col], row[effect_col]), axis=1)
        elif experiment_num == 2:
            if not put_in_passive:
                df['prompt'] = df.apply(lambda row: exp2_generate_prompt(row[cause_col], row[effect_col]), axis=1)
            else:
                df['prompt'] = df.apply(lambda row: exp2_passive_generate_prompt(row[cause_col], row[effect_col]), axis=1)
    df['method'] = ["POST" for _ in range(len(df))]
    df['url'] = ["/v1/chat/completions" for _ in range(len(df))]
    new_df = df.filter(['custom_id', 'method', 'url', 'prompt'])
    d = new_df.to_dict(orient='records')
    return d

def create_custom_ids(df, file_n):
    df['custom_id'] = ["request-" + str(i + 1) for i in range(len(df))]
    di = df.to_dict(orient='records')
    new_file_name = file_n.split(".xlsx")[0] + ".jsonl"
    with open(new_file_name, 'w') as f:
        for record in di:
            f.write(json.dumps(record) + '\n')
    return df

def process_body(prompt: str,
                 model: str):
    return {
        'model' : model,
        'messages' : [
            {
                "role": "user",
                "content": prompt
             }
        ],
        'store': True
    }

def create_batch_file(dictionary: dict,
                      model_name: str,
                      batch_file_name: str
                      ):
    for line in dictionary:
        line['body'] = process_body(prompt = line['prompt'], model = model_name)
        del line['prompt']

    directory = os.path.dirname(batch_file_name)
    # Create directories if they do not exist
    os.makedirs(directory, exist_ok=True)

    with open(batch_file_name, 'w') as f:
        for record in dictionary:
            f.write(json.dumps(record) + '\n')

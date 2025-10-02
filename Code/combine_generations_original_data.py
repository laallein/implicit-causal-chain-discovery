import json
import pandas as pd
import re


def process_chain(chain):
    c_ = [remove_unwanted_tokens(event.strip()) for event in chain]
    c = [e for e in c_ if
         (len(e) > 1 and e not in ['</chain>', '<chain>', '<chain'])]
    return {"chain": c,
            "chain_length": len(c),
            "causal_relations":[{'cause': c[i-1].strip(), 'effect': c[i].strip()} for i in range(1, len(c))]
    }


def remove_numbering(text):
    # Use regex to remove numbering at the beginning of each line
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
    return re.sub(r'\.', '', text)


def remove_unwanted_tokens(s: str):
    if "</step>" in s:
        s = s.replace("</step>", "")
    s = remove_parentheses(remove_numbering(s))
    if 'leads to' in s:
        s = s.split('leads to')[0]
    s = s.replace('→ ', '')
    return s

def contains_newline_followed_by_string(input_string):
    # Define the regular expression pattern
    pattern = r"\n(.+)"
    # Search for the pattern in the input string
    match = re.search(pattern, input_string)
    # Return True if the pattern is found, otherwise False
    return bool(match)


def remove_parentheses(text):
    # Use regex to remove parentheses and the text within them
    return re.sub(r'\(.*?\)', '', text)


def remove_empty_chains(causal_chain: list):
    # remove LLM intro
    if causal_chain[0]['chain_length'] == 1 and len(causal_chain[0]['causal_relations']) == 0:
        causal_chain = causal_chain[1:]
    # outro is already captured
    if causal_chain == []:
        return causal_chain
    if len(causal_chain[-1]['causal_relations']) == 0:
        causal_chain = causal_chain[:-1]
    # outro is part of effect
    if contains_newline_followed_by_string(causal_chain[-1]['causal_relations'][-1]['effect']):
        causal_chain[-1]['causal_relations'][-1]['effect'] = causal_chain[-1]['causal_relations'][-1]['effect'].split('\n')[0]
        causal_chain[-1]['chain'][-1] = causal_chain[-1]['chain'][-1].split('\n')[0]
    # remove empty chains
    cc = [chain_dict for chain_dict in causal_chain if len(chain_dict['chain']) > 0]
    for chain_info in cc:
        if '</chain>' in chain_info['chain'][-1]:
            chain_info['chain'][-1] = chain_info['chain'][-1].replace('</chain>', '').strip()
            chain_info['causal_relations'][-1]['effect'] = chain_info['causal_relations'][-1]['effect'].replace('</chain>', '').strip()
        #remove redundant intro from beginning of chain:
        print(chain_info['chain'])
        if contains_newline_followed_by_string(chain_info['chain'][0]):
            chain_info['chain'][0] = chain_info['chain'][0].split('\n')[-1].strip()
            chain_info['causal_relations'][0]['cause'] = chain_info['causal_relations'][0]['cause'].split('\n')[-1].strip()
        if contains_newline_followed_by_string(chain_info['chain'][-1]):
            chain_info['chain'][-1] = chain_info['chain'][-1].split('\n')[0].strip()
            chain_info['causal_relations'][-1]['effect'] = chain_info['causal_relations'][-1]['effect'].split('\n')[
                0].strip()
        chain_info['chain'] = [elem.strip().rstrip() for elem in chain_info['chain']]
    # chain needs to have at least a length of 2 steps
    cc = [chain for chain in cc if len(chain['chain']) > 2]
    return cc

def special_case_one_step(chains: list):
    for i, chain in enumerate(chains):
        if chain.count('<step>') == 1:
            chains[i] = chain.split('<step>')[0].replace('>', '<step>')
    return chains

def process_chains(c: str,
                   d: dict):
    if "</think>" in c:
        c = c.split("</think>", 1)[1]
    if "<chain>" not in c or c.count("<chain>") < 2:
        pattern = r"(?i)causal chain:|chain:|chain \d+:|causal chain \d+:|causal chain \d+|chain \d+"
        chains_not_split = [l.replace('*', '').replace('-', '').strip() for l in re.split(pattern, c)]
    else:
        # First remove `<chain>` and `<step>` from the intro text, e.g. separated nuy the token '<chain>'.
        c = re.sub(r'`<chain>`|`<step>`', '', c)
        chains_not_split = [l.replace('*', '').replace('-', '').strip() for l in c.split("<chain>")]
    chains_not_split = [c for c in chains_not_split if 'chains are separated by' not in c]
    chains_not_split = special_case_one_step(chains_not_split)
    all_causal_chains = [process_chain(chain.split('<step>')) for chain in chains_not_split
                              if chain.split('<step>') != ""]
    d['all_causal_chains'] = remove_empty_chains(all_causal_chains)
    d['num_causal_chains'] = count_chains(d['all_causal_chains'])
    nth_chain = 'causal_chain_'
    for n in range(d['num_causal_chains']):
        name = nth_chain + str(n)
        d[name] = " --> ". join(d['all_causal_chains'][n]['chain'])
    return d

def structure_chains_in_json_format(prediction_lines: list):
    new_list = list()
    length_reason = list()
    for line in prediction_lines:
        sub_dict = dict()
        sub_dict['custom_id'] = line['custom_id']
        if 'response' not in line.keys():
            if line['choices'][0]['finish_reason'] == 'length':
                length_reason.append(line['custom_id'])
                continue
        if 'response' not in line.keys():
            sub_dict = process_chains(line['choices'][0]['message']['content'], sub_dict)
        else:
            sub_dict = process_chains(line['response']['body']['choices'][0]['message']['content'], sub_dict)
        new_list.append(sub_dict)
    with open('length.json', 'w') as f:
        f.write(json.dumps(length_reason))
    return new_list

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def merge_jsonl_files(file1, file2, output_file):
    data1 = load_jsonl(file1)
    data2_ = load_jsonl(file2)

    data2 = structure_chains_in_json_format(prediction_lines = data2_)

    merged_data = {}

    for record in data1:
        custom_id = record.get('custom_id')
        if custom_id:
            merged_data[custom_id] = record

    for record in data2:
        custom_id = record.get('custom_id')
        if custom_id:
            if custom_id in merged_data:
                merged_data[custom_id].update(record)
            else:
                merged_data[custom_id] = record

    # Write the merged data to the output JSONL file
    with open(output_file, 'w') as f:
        for record in merged_data.values():
            f.write(json.dumps(record) + '\n')

    print(f"Merged data has been written to {output_file}")


def count_chains(l: list):
    return len([chain for chain in l if chain['chain'] != ['']])


def transform_to_csv(output_file):
    f = load_jsonl(output_file)
    df = pd.DataFrame(f)
    df.to_csv(output_file.replace('.jsonl', '.csv'), index=False)

def postprocess_chains(original_dataset_file: str,
                       generation_file: str,
                       output_file: str):
    merge_jsonl_files(file1 = original_dataset_file, file2 = generation_file, output_file = output_file)
    transform_to_csv(output_file)
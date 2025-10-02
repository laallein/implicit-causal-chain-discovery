import os

import pandas as pd
import json
"""
Experiment 1: Internal consistency of LLM. For each cause-and-effect relationship in the chain, let the model predict whether event t causes event t+1.
    + ask in passive voice, as most cause mention often precedes event mention, which can affect evaluation.

Experiment 2: Perturbation. Switch cause and effect, and conduct experiment 1 again.
"""
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def exp1_generate_prompt(cause, effect):
    return f"""Answer with 'yes' or 'no' only. Does {cause} cause {effect}?"""

def exp1_passive_generate_prompt(cause, effect):
    return f"""Answer with 'yes' or 'no' only. Is {effect} caused by {cause}?"""

def exp2_generate_prompt(cause, effect):
    return f"""Answer with 'yes' or 'no' only. Does {effect} cause {cause}?"""

def exp2_passive_generate_prompt(cause, effect):
    return f"""Answer with 'yes' or 'no' only. Is {cause} caused by {effect}?"""

def process_answer(c: str,
                   d: dict,
                   name: str):
    if '</think>' in c:
        c = c.split('</think>')[-1].strip()
    c = c.lower().replace('.', '')
    if len(c)  > 3:
        if c.startswith('yes'):
            c = 'yes'
        elif c.startswith('no'):
            c = 'no'
    d[name] = c
    return d

def extract_model_response(prediction_lines: list,
                           column_name: str):
    n_list = list()
    for line in prediction_lines:
        sub_dict = dict()
        sub_dict['custom_id'] = line['custom_id']
        if 'response' in line:
            sub_dict = process_answer(line['response']['body']['choices'][0]['message']['content'], sub_dict, column_name)
        else:
            sub_dict = process_answer(line['choices'][0]['message']['content'], sub_dict, column_name)
        n_list.append(sub_dict)
    return n_list

def merge_results(main_file, new_file, name_experiment):
    df = pd.read_csv(main_file)
    analysis_results = extract_model_response(load_jsonl(new_file), name_experiment)
    df2 = pd.DataFrame(analysis_results)
    combined_df = pd.merge(df, df2, on="custom_id")
    combined_df = combined_df[['custom_id'] + [col for col in combined_df.columns if col != 'custom_id']]
    return combined_df

def format_analysis_files(args, pol_num, passive):
    if args.experiment_num == 1:
        if not passive:
            batch_experiment_file = "1-Experiment/Polaris" + pol_num + "_causal_relations_analysis_exp1_about_" + args.under_assessment + ".jsonl"
            results_experiment_file = "Experiment1-LM-output/" + args.under_assessment + "/Polaris" + pol_num + "_exp1_" + args.conducting_assessment + "_about_" + args.under_assessment + ".jsonl"
        else:
            batch_experiment_file = "1-Experiment/Polaris" + pol_num + "_causal_relations_analysis_exp1_about_" + args.under_assessment + "_PASSIVE.jsonl"
            results_experiment_file = "Experiment1-LM-output/" + args.under_assessment + "/Polaris" + pol_num + "_exp1_" + args.conducting_assessment + "_about_" + args.under_assessment + "_PASSIVE.jsonl"
    elif args.experiment_num == 2:
        if not passive:
            batch_experiment_file = "2-Experiment/Polaris" + pol_num + "_causal_relations_analysis_exp2_about_" + args.under_assessment + ".jsonl"
            results_experiment_file = "Experiment2-LM-output/"  + args.under_assessment + "/Polaris" + pol_num + "_exp2_" + args.conducting_assessment + "_about_" + args.under_assessment + ".jsonl"
        else:
            batch_experiment_file = "2-Experiment/Polaris" + pol_num + "_causal_relations_analysis_exp2_about_" + args.under_assessment + "_PASSIVE.jsonl"
            results_experiment_file = "Experiment2-LM-output/" + args.under_assessment + "/Polaris" + pol_num + "_exp2_" + args.conducting_assessment + "_about_" + args.under_assessment + "_PASSIVE.jsonl"
    batch_file = args.analysis_batch_directory + args.under_assessment + "/" + batch_experiment_file
    output_file = args.analysis_results_directory + results_experiment_file
    return batch_file, output_file

def main():
    being_assessed = 'llama_3_70b'
    doing_assessment = 'mixtral'
    # Set dataset_num to 4 to run evaluation A4 (Cross-model evaluation of the causal chains) with PolarIs4CAUS and
    # to 3 to run with PolarIs3CAUS
    dataset_num = 4
    passive = False

    df = merge_results(
        main_file="../../Data/Polaris/Polaris" + str(dataset_num) + "_subrelations/Polaris" + str(dataset_num) + "_causal_sub-relations_" + being_assessed + ".csv",
        new_file="../../Analysis/CausalRelations/Experiment1-LM-output/" + being_assessed + "/Polaris" + str(dataset_num) + "_exp1_" + doing_assessment + "_about_" + being_assessed + ".jsonl",
        name_experiment="Does_cause_lead_to_effect")
    directory = os.path.dirname("../../Analysis/CausalRelations/All-Experiments-LM-output/" + being_assessed + "/Polaris" + str(dataset_num) + "_all_analyses_" + doing_assessment + "_about_" + being_assessed + ".csv")
    os.makedirs(directory, exist_ok=True)
    df.to_csv("../../Analysis/CausalRelations/All-Experiments-LM-output/" + being_assessed + "/Polaris" + str(dataset_num) + "_all_analyses_" + doing_assessment + "_about_" + being_assessed + ".csv", index=False)
    df = merge_results(
                  main_file="../../Analysis/CausalRelations/All-Experiments-LM-output/" + being_assessed + "/Polaris" + str(dataset_num) + "_all_analyses_" + doing_assessment + "_about_" + being_assessed + ".csv",
                  new_file="../../Analysis/CausalRelations/Experiment2-LM-output/" + being_assessed + "/Polaris" + str(dataset_num) + "_exp2_" + doing_assessment + "_about_" + being_assessed + ".jsonl",
                  name_experiment="Does_effect_lead_to_cause_SWITCH")
    df.to_csv("../../Analysis/CausalRelations/All-Experiments-LM-output/" + being_assessed + "/Polaris" + str(dataset_num) + "_all_analyses_" + doing_assessment + "_about_" + being_assessed + ".csv", index=False)

    if passive:
        df = merge_results(
            main_file="../../Analysis/CausalRelations/All-Experiments-LM-output/" + being_assessed + "/Polaris" + str(
                dataset_num) + "_all_analyses_" + doing_assessment + "_about_" + being_assessed + ".csv",
            new_file="../../Analysis/CausalRelations/Experiment1-LM-output/" + being_assessed + "/Polaris" + str(
                dataset_num) + "_exp1_" + doing_assessment + "_about_" + being_assessed + "_PASSIVE.jsonl",
            name_experiment="Does_cause_lead_to_effect_PASSIVE")
        df.to_csv("../../Analysis/CausalRelations/All-Experiments-LM-output/" + being_assessed + "/Polaris" + str(
            dataset_num) + "_all_analyses_" + doing_assessment + "_about_" + being_assessed + ".csv", index=False)
        df = merge_results(
            main_file="../../Analysis/CausalRelations/All-Experiments-LM-output/" + being_assessed + "/Polaris" + str(
                dataset_num) + "_all_analyses_" + doing_assessment + "_about_" + being_assessed + ".csv",
            new_file="../../Analysis/CausalRelations/Experiment2-LM-output/" + being_assessed + "/Polaris" + str(
                dataset_num) + "_exp2_" + doing_assessment + "_about_" + being_assessed + "_PASSIVE.jsonl",
            name_experiment="Does_effect_lead_to_cause_SWITCH_PASSIVE")
        df.to_csv("../../Analysis/CausalRelations/All-Experiments-LM-output/" + being_assessed + "/Polaris" + str(
            dataset_num) + "_all_analyses_" + doing_assessment + "_about_" + being_assessed + ".csv", index=False)



if __name__ == '__main__':
    main()
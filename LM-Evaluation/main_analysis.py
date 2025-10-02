from statistics import stdev, mean
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict
from itertools import combinations
from scipy.spatial import distance
import os


def flatten(xss):
    return [x for xs in xss for x in xs]

def overall_correctness_chains(c_list_fce: list):
    return sum([a.count(True) for a in c_list_fce]), sum([a.count(False) for a in c_list_fce])

def breaking_point_chains(breaking_points: list):
    points = flatten(breaking_points)
    points_incorrect = [a for a in points if a != -1.0]
    beginning = [a for a in points_incorrect if a < 0.4]
    middle = [a for a in points_incorrect if 0.4 <= a <= 0.6]
    end = [a for a in points_incorrect if a > 0.6]
    return mean(points_incorrect), stdev(points_incorrect), len(beginning), len(middle), len(end)

def correlation_cov_pearsonr(a: list,
                             b: list):
    covariance_ = compute_covariance(a, b)
    pearson_corr, p = compute_pearson_corr(a, b)
    return covariance_, pearson_corr, p

def compute_covariance(a: list,
                       b: list):
    return np.cov(a, b)

def compute_pearson_corr(a: list,
                         b: list):
    rho, p = pearsonr(a, b)
    return rho, p

def extract_chains(a: list,
                   i: str):
    ce_sub_relations = [ce for ce in a if ce['custom_id'].startswith(i+'-')]
    for ce in ce_sub_relations:
        ce['custom_id'] = ce['custom_id'].replace(i + '-', '')
    num_chains = sorted(set([ce['custom_id'].split('-')[0] for ce in ce_sub_relations]))
    chains = [list() for _ in range(len(num_chains))]
    for ce in ce_sub_relations:
        chains[int(ce['custom_id'].split('-')[0])].append(ce)
    return chains

def fully_causal_chains(c:list):
    cause_effect_list = [k['Does_cause_lead_to_effect'] for k in c]
    effect_cause_list = [k['Does_effect_lead_to_cause_SWITCH'] for k in c]
    res = {'length_chain': len(c),
           'fully_c_to_e': True,
           'num_correct_c_to_e': len(cause_effect_list),
           'num_violations_c_to_e': 0.0,
           'ratio_correct_c_to_e': 1.0,
           'position_first_break_c_to_e': -1.0,
           'fully_no_e_to_c': True,
           'num_correct_no_e_to_c': len(effect_cause_list),
           'num_violations_no_e_to_c': 0.0,
           'ratio_correct_no_e_to_c': 1.0,
           'position_first_break_no_e_to_c': -1.0,
           'fully_ce_no_ec': False
           }
    if cause_effect_list.count('yes') < len(c):
        res['fully_c_to_e'] = False
        res['num_correct_c_to_e'] = cause_effect_list.count('yes')
        res['num_violations_c_to_e'] = len(cause_effect_list) - res['num_correct_c_to_e']
        res['ratio_correct_c_to_e'] = cause_effect_list.count('yes')/len(c)
        for i, elem in enumerate(cause_effect_list):
            if elem != 'yes':
                res['position_first_break_c_to_e'] = i/len(cause_effect_list)
                break
    if effect_cause_list.count('no') < len(c):
        res['fully_no_e_to_c'] = False
        res['num_correct_no_e_to_c'] = effect_cause_list.count('no')
        res['num_violations_no_e_to_c'] = len(effect_cause_list) - res['num_correct_no_e_to_c']
        res['ratio_correct_no_e_to_c'] = effect_cause_list.count('no')/len(c)
        for i, elem in enumerate(effect_cause_list):
            if elem != 'yes':
                res['position_first_break_no_e_to_c'] = i/len(effect_cause_list)
                break
    if res['fully_c_to_e'] and res['fully_no_e_to_c']:
        res['fully_ce_no_ec'] = True
    return res

def compute_histogram(a: list, discrete: bool, density: bool, mask=False):
    if discrete:
        bins = [i for i in range(1, max(a) + 1)]
    else:
        bins = [0.0,0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1.0]
    if mask:
        a = [e for e in a if e != -1.0]
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    return np.histogram(a, bins=bins, density=density)

def align_num_len(num_list: list,
                  len_list: list):
    new_num_list = list()
    for i, l in enumerate(len_list):
        new_num_list.append([num_list[i]] * len(l))
    return new_num_list

def perform_general_analysis(chain_dict: dict):
    alignment_num_len_chains = align_num_len(chain_dict['num_chains'], chain_dict['length_chain'])
    co, rho, p = correlation_cov_pearsonr(flatten(chain_dict['length_chain']), flatten(alignment_num_len_chains))
    analysis_text=f"""
    Total number of original C-E relations: {len(chain_dict['num_chains'])}
    Total number of generated chains: {sum(chain_dict['num_chains'])}
    -- Mean per C-E relation: {mean(chain_dict['num_chains'])}; Std: {stdev(chain_dict['num_chains'])}; min: {min(chain_dict['num_chains'])}; max: {max(chain_dict['num_chains'])}
    -- Distribution - histogram: 
    {compute_histogram(chain_dict['num_chains'], discrete=True, density=False)}
    {compute_histogram(chain_dict['num_chains'], discrete=True, density=True)}
    Length of generated chains:
    -- Mean: {mean(flatten(chain_dict['length_chain']))}; Std: {stdev(flatten(chain_dict['length_chain']))}; min: {min(flatten(chain_dict['length_chain']))}; max: {max(flatten(chain_dict['length_chain']))}
    -- Distribution - histogram: 
    {compute_histogram(flatten(chain_dict['length_chain']), discrete=True, density=False)}
    {compute_histogram(flatten(chain_dict['length_chain']), discrete=True, density=True)}
    Correlation between number of generated chains and length of generated chains:
    {co} (covariance), {rho, p} (pearson r)
"""
    print(analysis_text)

def perform_experiment_analyses(chain_dict: dict,
                                num_experiment: int):
    if num_experiment == 1:
        num_correct, num_incorrect = overall_correctness_chains(chain_dict['fully_c_to_e'])
        mean_breaking, std_breaking, b, m, e = breaking_point_chains(chain_dict['position_first_break_c_to_e'])
        bp = chain_dict['position_first_break_c_to_e']
        cov_len_corr, rho_len_corr, p_len_corr = correlation_cov_pearsonr(flatten(chain_dict['length_chain']), flatten(chain_dict['ratio_correct_c_to_e']))
        cov_num_chains_ratio, rho_num_chains_ratio, p_num_chains_ratio = correlation_cov_pearsonr(chain_dict['num_chains'], chain_dict['ratio_fully_c_to_e'])
    elif num_experiment == 2:
        num_correct, num_incorrect = overall_correctness_chains(chain_dict['fully_no_e_to_c'])
        mean_breaking, std_breaking, b, m, e = breaking_point_chains(chain_dict['position_first_break_no_e_to_c'])
        bp = chain_dict['position_first_break_no_e_to_c']
        cov_len_corr, rho_len_corr, p_len_corr = correlation_cov_pearsonr(flatten(chain_dict['length_chain']),
                                                                          flatten(chain_dict['ratio_correct_no_e_to_c']))
        cov_num_chains_ratio, rho_num_chains_ratio, p_num_chains_ratio = correlation_cov_pearsonr(
            chain_dict['num_chains'], chain_dict['ratio_fully_no_e_to_c'])

    analysis_text = f"""
    total number of "correct" chains: {num_correct} / {num_correct/(num_correct+num_incorrect)} %
    total number of "incorrect" chains: {num_incorrect} / {num_incorrect/(num_correct+num_incorrect)} %
    Correlation between:
        - length of chain and correctness? {cov_len_corr} (covariance), {rho_len_corr, p_len_corr} (pearson r)
        - number of generated chains and ratio correct chains? {cov_num_chains_ratio} (covariance), {rho_num_chains_ratio, p_num_chains_ratio} (pearson r)
    About INCORRECT chains:
    -- When does the chain start to break? (first occurrence of False in incorrect chains): 
       {mean_breaking} (mean); {std_breaking} (std)
       {b} ({b/num_incorrect}%) in the beginning of the chain, {m} ({m/num_incorrect}%) in the middle of the chain, {e} ({e/num_incorrect}%) in the end of the chain.
       Histograms:
       {compute_histogram(flatten(bp), discrete=False, density=False, mask=True)}
       {compute_histogram(flatten(bp), discrete=False, density=True, mask=True)}
    
    """

    print(analysis_text)

def transform_to_boolean(l):
    new_l = list()
    for elem in l:
        if elem == 'yes':
            new_l.append(1)
        else:
            new_l.append(0)
    return new_l

def retrieve_chain_info_passive(original_data_file: str,
                                analysis_file: str):
    analyses = pd.read_csv(original_data_file).to_dict(orient="records")
    analysis_exp1 = transform_to_boolean([e['Does_cause_lead_to_effect_PASSIVE'] for e in analyses])
    correct_subrels_exp1 = analysis_exp1.count(1)
    analysis_exp2 = transform_to_boolean([e['Does_effect_lead_to_cause_SWITCH_PASSIVE'] for e in analyses])
    correct_subrels_exp2 = analysis_exp2.count(0)
    original_data = pd.read_csv(analysis_file).to_dict(orient="records")
    all_chains = defaultdict(list)
    counter = 0
    for original_ce in original_data:
        counter +=1
        chains = extract_chains(analyses, original_ce['custom_id'])
        all_chains_ce_rel = {
            'custom_id': original_ce['custom_id'],
            'length_chain': list(),
           'fully_c_to_e': list(),
           'num_correct_c_to_e': list(),
            'num_violations_c_to_e': list(),
           'ratio_correct_c_to_e':list(),
           'position_first_break_c_to_e': list(),
           'fully_no_e_to_c': list(),
           'num_correct_no_e_to_c': list(),
            'num_violations_no_e_to_c': list(),
           'ratio_correct_no_e_to_c': list(),
           'position_first_break_no_e_to_c': list(),
           'fully_ce_no_ec': list()
           }
        for chain in chains:
            r = fully_causal_chains(chain)
            for key, value in r.items():
                all_chains_ce_rel[key].append(value)

        all_chains_ce_rel['num_chains'] = len(all_chains_ce_rel['length_chain'])
        if all_chains_ce_rel['num_chains'] == 0:
            print(counter)
            continue
        all_chains_ce_rel['average_chain_length'] = sum(all_chains_ce_rel['length_chain'])/all_chains_ce_rel['num_chains']
        all_chains_ce_rel['total_fully_c_to_e'] = all_chains_ce_rel['fully_c_to_e'].count(True)
        all_chains_ce_rel['ratio_fully_c_to_e'] = all_chains_ce_rel['total_fully_c_to_e']/all_chains_ce_rel['num_chains']
        all_chains_ce_rel['total_fully_no_e_to_c'] = all_chains_ce_rel['fully_no_e_to_c'].count(True)
        all_chains_ce_rel['ratio_fully_no_e_to_c'] = all_chains_ce_rel['total_fully_no_e_to_c'] / all_chains_ce_rel[
            'num_chains']
        all_chains_ce_rel['total_fully_correct'] = all_chains_ce_rel['fully_ce_no_ec'].count(True)
        all_chains_ce_rel['ratio_fully_correct'] = all_chains_ce_rel['total_fully_correct'] / all_chains_ce_rel[
            'num_chains']
        for k, v in all_chains_ce_rel.items():
            all_chains[k].append(v)

    return all_chains, analysis_exp1, analysis_exp2, correct_subrels_exp1, correct_subrels_exp2


def retrieve_chain_info(original_data_file: str,
                        analysis_file: str):
    analyses = pd.read_csv(original_data_file).to_dict(orient="records")
    analysis_exp1 = transform_to_boolean([e['Does_cause_lead_to_effect'] for e in analyses])
    correct_subrels_exp1 = analysis_exp1.count(1)
    analysis_exp2 = transform_to_boolean([e['Does_effect_lead_to_cause_SWITCH'] for e in analyses])
    correct_subrels_exp2 = analysis_exp2.count(0)
    original_data = pd.read_csv(analysis_file).to_dict(orient="records")
    all_chains = defaultdict(list)
    counter = 0
    for original_ce in original_data:
        counter +=1
        chains = extract_chains(analyses, original_ce['custom_id'])
        all_chains_ce_rel = {
            'custom_id': original_ce['custom_id'],
            'length_chain': list(),
           'fully_c_to_e': list(),
           'num_correct_c_to_e': list(),
            'num_violations_c_to_e': list(),
           'ratio_correct_c_to_e':list(),
           'position_first_break_c_to_e': list(),
           'fully_no_e_to_c': list(),
           'num_correct_no_e_to_c': list(),
            'num_violations_no_e_to_c': list(),
           'ratio_correct_no_e_to_c': list(),
           'position_first_break_no_e_to_c': list(),
           'fully_ce_no_ec': list()
           }
        for chain in chains:
            r = fully_causal_chains(chain)
            for key, value in r.items():
                all_chains_ce_rel[key].append(value)

        all_chains_ce_rel['num_chains'] = len(all_chains_ce_rel['length_chain'])
        if all_chains_ce_rel['num_chains'] == 0:
            print(counter)
            continue
        all_chains_ce_rel['average_chain_length'] = sum(all_chains_ce_rel['length_chain'])/all_chains_ce_rel['num_chains']
        all_chains_ce_rel['total_fully_c_to_e'] = all_chains_ce_rel['fully_c_to_e'].count(True)
        all_chains_ce_rel['ratio_fully_c_to_e'] = all_chains_ce_rel['total_fully_c_to_e']/all_chains_ce_rel['num_chains']
        all_chains_ce_rel['total_fully_no_e_to_c'] = all_chains_ce_rel['fully_no_e_to_c'].count(True)
        all_chains_ce_rel['ratio_fully_no_e_to_c'] = all_chains_ce_rel['total_fully_no_e_to_c'] / all_chains_ce_rel[
            'num_chains']
        all_chains_ce_rel['total_fully_correct'] = all_chains_ce_rel['fully_ce_no_ec'].count(True)
        all_chains_ce_rel['ratio_fully_correct'] = all_chains_ce_rel['total_fully_correct'] / all_chains_ce_rel[
            'num_chains']
        for k, v in all_chains_ce_rel.items():
            all_chains[k].append(v)

    return all_chains, analysis_exp1, analysis_exp2, correct_subrels_exp1, correct_subrels_exp2

def model_specific_assessment(og_file, a_file, passive = False, being_assessed=""):
    chains_dict, exp1, exp2, correct_subrels_1, correct_subrels_2 = retrieve_chain_info(original_data_file=og_file, analysis_file=a_file)
    print("*** General Analysis ***")
    perform_general_analysis(chains_dict)
    print("*** Experiment 1: does CAUSE cause EFFECT***")
    print("Total subrels 'yes': ", str(correct_subrels_1))
    perform_experiment_analyses(chains_dict, num_experiment=1)
    print("*** Experiment 2 (switch): does EFFECT cause CAUSE***")
    print("Total subrels 'no': ", str(correct_subrels_2))
    perform_experiment_analyses(chains_dict, num_experiment=2)
    if passive and being_assessed + "_about" in og_file:
        print("*** TEMPORAL CONTINGENCY: PASSIVE in prompts ***")
        chains_dict, exp1_pas, exp2_pas, correct_subrels_1_pas, correct_subrels_2_pas = retrieve_chain_info_passive(original_data_file=og_file, analysis_file=a_file)
        print("EXP 1: match active with passive")
        evaluate_agreement([exp1, exp1_pas])
        print("EXP 2: match active with passive")
        evaluate_agreement([exp2, exp2_pas])
    return chains_dict, exp1, exp2


def jaccard_distance(l1, l2):
    return distance.jaccard(l1, l2)

def hamming_distance(l1, l2):
    return distance.hamming(l1, l2)

def pairwise_comparison(lists: list):
    pairs = combinations(lists, 2)

    pairwise_agreements = []
    pairwise_disagreements = []

    for pair in pairs:
        pairwise_agreements.append(np.all(pair[0] == pair[1]))
        pairwise_disagreements.append(np.any(pair[0] != pair[1]))

    return pairwise_agreements, pairwise_disagreements

def evaluate_agreement(ds):
    jd = jaccard_distance(ds[0], ds[1])
    hd = hamming_distance(ds[0], ds[1])

    analysis_text = f"""
    Jaccard Distance (matching 1s): {jd}
    ---> Jaccard Similarity (matching 1s): {1-jd}
    Hamming Distance (proportion disagreeing components): {hd}
    ---> Hamming Similarity (proportion agreeing components): {1-hd}
"""
    print(analysis_text)

# def compare_chains(d1, d2):

def get_all_files(directory):
    # List all files in the given directory
    return [directory + f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def extract_ids_with_valid_chain(chain_dict):
    ids = chain_dict['custom_id']
    num_valid_chains = chain_dict['total_fully_correct']
    positions_valid_chains = chain_dict['fully_ce_no_ec']
    ratio_a1 = chain_dict['ratio_correct_c_to_e']
    ratio_a2 = chain_dict['ratio_correct_no_e_to_c']

    valid_chains = {'id': [], 'length': []}
    invalid_chains = {'id': [], 'length': []}
    valid_ce_pair = list()
    counter_valid = 0
    for i, ce_pair_id in enumerate(ids):
        if num_valid_chains[i] > 0:
            counter_valid += 1
            valid = []
            valid_length = []
            invalid = []
            invalid_length = []
            valid_ce_pair.append(ce_pair_id)
            for j, chain_valid in enumerate(positions_valid_chains[i]):
                if chain_valid: # and chain_dict['length_chain'][i][j] > 3:
                    valid.append(ce_pair_id + '-' + str(j))
                    valid_length.append(chain_dict['length_chain'][i][j])
                else:
                    if ratio_a1[i][j] < 1.0 and ratio_a2[i][j] < 1.0:
                        invalid.append(ce_pair_id + '-' + str(j))
                        invalid_length.append(chain_dict['length_chain'][i][j])
            valid_chains['id'].extend(valid)
            valid_chains['length'].extend(valid_length)
            invalid_chains['id'].extend(invalid)
            invalid_chains['length'].extend(invalid_length)

    print("total CE pairs with at least one fully correct chain: ", counter_valid)
    return valid_chains, invalid_chains, valid_ce_pair

def sort_in_descending_order(data):
    combined = list(zip(data['id'], data['score'], data['length']))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    sorted_ids, sorted_scores, sorted_lengths = zip(*sorted_combined)
    return {'id': sorted_ids, 'score': sorted_scores, 'length': sorted_lengths}


def main():
    # Get statistics (Table 3 in paper) and
    # Consistency with Jaccard dissimilarity and Hamming distance (A3)
    being_assessed = 'llama_3_70b'
    # Set dataset_num to 3 to run with PolarIs3CAUS data and to 4 to run with PolarIs4CAUS data
    dataset_num = 4
    passive = False

    dataset_name = "Polaris" + str(dataset_num)

    directory = "CausalRelations/All-Experiments-LM-output/" + being_assessed + "/"
    files = [f for f in get_all_files(directory) if dataset_name in f]
    a_file = "../Data/Polaris/" + dataset_name + "_full_with_chains_" + being_assessed + ".csv"

    print("CHAINS BY " + being_assessed)
    print('----------------')
    all_exp_1 = list()
    all_exp_2 = list()
    scoring_valid = {'id': list(), 'length': list(), 'score': list()}
    scoring_invalid = {'id': list(), 'length': list(), 'score': list()}
    valid_ce_pair = list()
    for f in files:
        print("EVAL BY: " + f)
        chains_dict, exp1, exp2 = model_specific_assessment(og_file=f, a_file=a_file, passive=passive, being_assessed=being_assessed)
        all_exp_1.append(exp1)
        all_exp_2.append(exp2)
        print("*****************")
        valid_chains, invalid_chains, valid_ce_pairs_list = extract_ids_with_valid_chain(chains_dict)
        valid_ce_pair.extend(valid_ce_pairs_list)
        for i, chain_id in enumerate(valid_chains['id']):
            if chain_id not in scoring_valid['id']:
                scoring_valid['id'].append(chain_id)
                scoring_valid['length'].append(valid_chains['length'][i])
                scoring_valid['score'].append(1)
            else:
                scoring_valid['score'][scoring_valid['id'].index(chain_id)] += 1
        for i, chain_id in enumerate(invalid_chains['id']):
            if chain_id not in scoring_invalid['id']:
                scoring_invalid['id'].append(chain_id)
                scoring_invalid['length'].append(invalid_chains['length'][i])
                scoring_invalid['score'].append(1)
            else:
                scoring_invalid['score'][scoring_invalid['id'].index(chain_id)] += 1

    sorted_valid = sort_in_descending_order(scoring_valid)
    df = pd.DataFrame(sorted_valid)
    df_sorted = df.sort_values(by='score', ascending=False)
    df_sorted.to_csv(dataset_name + '_valid_chains_' + being_assessed + ".csv", index=False)

    sorted_invalid = sort_in_descending_order(scoring_invalid)
    df = pd.DataFrame(sorted_invalid)
    df_sorted = df.sort_values(by='score', ascending=False)
    df_sorted.to_csv(dataset_name + '_invalid_chains_' + being_assessed + ".csv", index=False)

    print("*** AGREEMENT experiment 1 ***")
    evaluate_agreement(ds=all_exp_1)
    print("*** AGREEMENT experiment 2 ***")
    evaluate_agreement(ds=all_exp_2)

    print("Total unique chains that have been evaluated as fully correct by at least one LLM: ", len(df_sorted))
    print("For # CE pairs: ", len(set(valid_ce_pair)))
    print("Total unique chains that have been evaluated as fully correct by ALL LLMs: ", df_sorted[df['score'] == 7])

main()
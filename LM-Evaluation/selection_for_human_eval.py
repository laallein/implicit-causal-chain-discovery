import pandas as pd

def process_file(f, valid=True):
    df = pd.read_csv(f)
    print(len(df))
    df[['CE_pair', 'suffix']] = df['id'].str.rsplit('-', n=1, expand=True)
    unique_ce_overall = df['CE_pair'].unique()
    print(' -- for # CE pairs: ', len(unique_ce_overall))
    four_or_more_steps = df[df['length'] > 3]
    print(len(four_or_more_steps))
    four_or_more_steps[['CE_pair', 'suffix']] = four_or_more_steps['id'].str.rsplit('-', n=1, expand=True)
    unique_ce_four = four_or_more_steps['CE_pair'].unique()
    print(' -- for # CE pairs: ', len(unique_ce_four))
    valid_by_at_least_three_lms = four_or_more_steps[four_or_more_steps['score'] > 2]
    print(len(valid_by_at_least_three_lms))
    valid_by_at_least_three_lms[['CE_pair', 'suffix']] = valid_by_at_least_three_lms['id'].str.rsplit('-', n=1,
                                                                                                      expand=True)

    unique_ce_pairs = valid_by_at_least_three_lms['CE_pair'].unique()
    unique_ce_pairs.sort()
    print(' -- for # CE pairs: ', len(unique_ce_pairs))

    if not valid:
        unique_ce_overall.sort()
        return unique_ce_four, four_or_more_steps
    else:
        return unique_ce_pairs, valid_by_at_least_three_lms


def get_valid_invalid_chains(f, f2):
    unique_valid, df_valid = process_file(f)
    unique_invalid, df_invalid = process_file(f2, valid=False)

    print('*******')
    pairable_with_invalid_chain = list()
    for elem in unique_valid:
        if elem in unique_invalid:
            pairable_with_invalid_chain.append(elem)
    final_valid = {'id': [], 'score': [], 'length': [], 'CE_pair': []}
    final_invalid = {'id': [], 'score': [], 'length': [], 'CE_pair': []}

    by_length = {4: [], 5: [], 6: []}

    for elem in pairable_with_invalid_chain:
        candidate_valid_chains = df_valid[df_valid['CE_pair'] == elem]
        candidate_invalid_chains = df_invalid[df_invalid['CE_pair'] == elem]
        if len(candidate_valid_chains) > 1:
            max_score = candidate_valid_chains['score'].max()
            candidate_valid_chains = candidate_valid_chains[candidate_valid_chains['score'] == max_score]
            if len(candidate_valid_chains) > 1:
                candidate_valid_chains['suffix'] = candidate_valid_chains['suffix'].astype(int)
                min_score = candidate_valid_chains['suffix'].min()
                candidate_valid_chains = candidate_valid_chains[candidate_valid_chains['suffix'] == min_score]
        if len(candidate_invalid_chains) > 1:
            max_score = candidate_invalid_chains['score'].max()
            candidate_invalid_chains = candidate_invalid_chains[candidate_invalid_chains['score'] == max_score]
            if len(candidate_invalid_chains) > 1:
                length_valid = candidate_valid_chains['length']
                candidate_invalid_chains['abs_diff'] = (candidate_invalid_chains['length'] - length_valid.iloc[0]).abs()
                min_distance = candidate_invalid_chains['abs_diff'].min()
                candidate_invalid_chains = candidate_invalid_chains[
                    candidate_invalid_chains['abs_diff'] == min_distance]
                if len(candidate_invalid_chains) > 1:
                    candidate_invalid_chains['suffix'] = candidate_invalid_chains['suffix'].astype(int)
                    min_score = candidate_invalid_chains['suffix'].min()
                    candidate_invalid_chains = candidate_invalid_chains[candidate_invalid_chains['suffix'] == min_score]
        print("+++++++++++++++++++++++")
        print(candidate_valid_chains)
        print(candidate_invalid_chains)
        by_length[candidate_valid_chains['length'].iloc[0].astype(int)].append(
            candidate_valid_chains['CE_pair'].iloc[0])
        for key in final_valid.keys():
            final_valid[key].append(candidate_valid_chains[key].iloc[0])
            final_invalid[key].append(candidate_invalid_chains[key].iloc[0])

    return by_length, final_valid, final_invalid

def divide_in_equal_groups(l):
    for length in l.keys():
        k, m = divmod(len(l[length]), 3)
        split_list = [l[length][i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(3)]
        print(split_list)

def save_to_csv(df_o1, o1_valid, o1_invalid, f_name):
    chains = {'id_val': [], 'id_inval': [], 'chain_val': [], 'chain_inval': []}
    for vi, ii in zip(o1_valid['id'], o1_invalid['id']):
        filtered_valid = df_o1[df_o1['custom_id'].str.startswith(vi)]
        causes = filtered_valid['cause'].tolist()
        effects = filtered_valid['effect'].tolist()
        chain_valid = causes + [effects[-1]]
        chains['id_val'].append(vi)
        chains['chain_val'].append(' --> '.join(chain_valid))

        filtered_invalid = df_o1[df_o1['custom_id'].str.startswith(ii)]
        causes = filtered_invalid['cause'].tolist()
        effects = filtered_invalid['effect'].tolist()
        chain_invalid = causes + [effects[-1]]
        chains['id_inval'].append(ii)
        chains['chain_inval'].append(' --> '.join(chain_invalid))

    df = pd.DataFrame.from_dict(chains)
    df.to_csv(f_name, index=False)

def main():
    o1_length, o1_valid, o1_invalid = get_valid_invalid_chains(f = 'Polaris4_valid_chains_o1.csv', f2 = 'Polaris4_invalid_chains_o1.csv')
    mini_length, mini_valid, mini_invalid = get_valid_invalid_chains(f = 'Polaris4_valid_chains_o1-mini.csv', f2 = 'Polaris4_invalid_chains_o1-mini.csv')
    print("*** O1 ***")
    print("# pairs: ", len(o1_valid['id']))
    divide_in_equal_groups(o1_length)
    print(o1_valid)
    print(o1_invalid)
    print("*** O1-mini ***")
    print("# pairs: ", len(mini_valid['id']))
    divide_in_equal_groups(mini_length)
    print(mini_valid)
    print(mini_invalid)

    df_o1 = pd.read_csv('CausalRelations/All-Experiments-LM-output/o1/Polaris4_all_analyses_gpt4o_about_o1.csv')
    df_mini = pd.read_csv('CausalRelations/All-Experiments-LM-output/o1-mini/Polaris4_all_analyses_gpt4o_about_o1-mini.csv')
    save_to_csv(df_o1, o1_valid, o1_invalid, f_name="human_eval_o1.csv")
    save_to_csv(df_mini, mini_valid, mini_invalid, f_name="human_eval_o1-mini.csv")

if __name__ == "__main__":
    main()
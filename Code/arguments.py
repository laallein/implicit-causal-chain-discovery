import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="preprocess and postprocess files for causal chain generation")

    # Task arguments

    parser.add_argument("--generate_chains", type=bool, required=False, help="Ask model to generate chains")
    parser.add_argument("--analyze_chains", type=bool, required=False, help="Ask model to analyze chains")

    # File arguments

    parser.add_argument(
        "--dataset_file", type=str, default="../../Data/Polaris/PolarIs3_CAUS_NPR_v2.xlsx"
    )
    parser.add_argument(
        "--dataset_file_jsonl", type=str, default="../../Data/Polaris/PolarIs3_CAUS_NPR_v2.jsonl"
    )
    parser.add_argument(
        "--batch_file", type=str, default="../../Data/Polaris/BatchFiles/Polaris3_openai_batch_file_o1-mini.jsonl"
    )
    parser.add_argument(
        "--generated_CC_file", type=str, default="GeneratedCausalChains/Polaris3_phi_4-mini.jsonl"
    )
    parser.add_argument(
        "--full_dataset_with_chains", type=str, default="../../Data/Polaris/Polaris3_full_with_chains_phi_4-mini.jsonl"
    )

    #OpenAI arguments

    parser.add_argument(
        "--model_openai", type=str, default="o1-mini-2024-09-12", choices=["o1-preview-2024-09-12", "o1-mini-2024-09-12", "gpt-4o-2024-11-20"]
    )
    parser.add_argument(
        "--openai_batch_id", type=str, default=""
    )
    parser.add_argument(
        "--openai_output_file_id", type=str, default=""
    )

    #LLama arguments
    parser.add_argument(
        "--model_llama", type=str, default="llama3.2"
    )
    parser.add_argument(
        "--submit_to_llama", type=bool, required=False, help="Use LLama for generation"
    )

    #Deepseek arguments
    parser.add_argument(
        "--submit_to_deepseek", type=bool, required=False, help="Use DeepSeek for generation"
    )

    #Nemotron arguments
    parser.add_argument(
        "--submit_to_nemotron", type=bool, required=False, help="Use Nemotron for generation"
    )

    #Mixtral arguments
    parser.add_argument(
        "--submit_to_mixtral", type=bool, required=False, help="Use Mixtral for generation"
    )
    # Processing arguments

    #Qwq arguments
    parser.add_argument(
        "--submit_to_qwq", type=bool, required=False, help="Use QWQ for generation"
    )

    #Phi-4-mini
    parser.add_argument(
        "--submit_to_phi", type=bool, required=False, help="Use Phi for generation"
    )

    parser.add_argument(
        "--submit_to_mistral_nemo", type=bool, required=False, help="Use mistral nemo for generation"
    )

    parser.add_argument(
        "--submit_to_llama_3", type=bool, required=False, help="Use llama 3 for generation"
    )

    parser.add_argument(
        "--preprocessing", type=bool, default=False, help="Preprocessing of new dataset file for input "
                                                         "to OpenAI"
    )
    parser.add_argument(
        "--submit_to_openai", type=bool, default=False, help="Submit batch for OpenAI API"
    )
    parser.add_argument(
        "--postprocessing", type=bool, default=False, help="Retrieval and postprocessing of generated "
                                                          "causal chains from OpenAI API"
    )
    parser.add_argument(
        "--cause_column", type=str, default="Aggregated_cause v2.0", choices=["Aggregated_cause v2.0"]
    )
    parser.add_argument(
        "--effect_column", type=str, default="Aggregated_effect v2.0", choices=["Aggregated_effect v2.0"]
    )

    # Dataset analysis arguments

    parser.add_argument(
        "--sub_causal_relations_file", type=str, default="../../Data/Polaris/Polaris4_subrelations/Polaris4_causal_sub-relations_o1-mini.csv"
    )
    parser.add_argument(
        "--analysis_batch_file", type=str, default="../../Data/Polaris/BatchFiles_analyses/o1/1-Experiment/Polaris4_causal_relations_analysis_exp1_about_o1.jsonl"
    )
    parser.add_argument(
        "--experiment_num", type=int, default=1, help="Number of experiment you are analyzing", choices=[1,2]
    )
    parser.add_argument(
        "--under_assessment", type=str, default="deepseek_r1", choices=["o1", "o1-mini", "gpt4o", "deepseek_r1",
                                                               "llama31_nemotron", "llama_3_70b", "mistral_nemo", "mixtral", "phi_4-mini", "qwq_32b"]
    )
    parser.add_argument(
        "--conducting_assessment", type=str, default="deepseek_r1", choices=["o1", "o1-mini", "gpt4o", "deepseek_r1",
                                                               "llama31_nemotron", "llama_3_70b", "mistral_nemo",
                                                               "mixtral", "phi_4-mini", "qwq_32b"]
    )
    parser.add_argument(
        "--analysis_batch_directory", type=str,
        default="../../Data/Polaris/BatchFiles_analyses/"
    )
    parser.add_argument(
        "--analysis_results_directory", type=str,
        default="../../Analysis/CausalRelations/"
    )
    parser.add_argument(
        "--analysis_results_file", type=str, default="../../Analysis/CausalRelations/Experiment1-LM-output/phi_4-mini/Polaris4_exp1_o1-mini_about_phi_4-mini.jsonl"
    )
    parser.add_argument(
        "--passive", default=False, help="put analysis prompt in passive"
    )
    parser.add_argument(
        "--polaris", default=4, help="number of polaris dataset the experiments are conducted on"
    )

    args = parser.parse_args()
    return args
from arguments import parse_args
from prepare_batch_file import process_input_file, create_batch_file
from upload_batch_file_to_openai import upload_to_openai, upload_to_lm, generate_with_llama, generate_with_deepseek, generate_with_nemotron, generate_with_mixtral, generate_with_qwq, generate_with_phi, generate_with_mistral_nemo, generate_with_llama_3
from retrieve_batch_generation_results import retrieve_from_openai
from combine_generations_original_data import postprocess_chains
from analyze_chains import format_analysis_files
import json

def main():
    args = parse_args()
    args.polaris = str(args.polaris)

    # Set generate_chains = True for running the causal chain inference experiments; Section "Causal Chain Inference"
    # Includes preprocessing, inference, and postprocessing (see detailed outline of postprocessing steps in the
    # Supplementary Material.
    if args.generate_chains:

        if args.preprocessing:
            f = process_input_file(file_name=args.dataset_file,
                                cause_col=args.cause_column,
                                effect_col=args.effect_column)
            create_batch_file(dictionary=f,
                            model_name=args.model_openai,
                            batch_file_name=args.batch_file)

        if args.submit_to_openai and not args.postprocessing:
            upload_to_openai(file_to_be_uploaded=args.batch_file)

        elif args.submit_to_llama:
            generate_with_llama(file_with_prompts=args.batch_file)

        elif args.submit_to_deepseek:
            generate_with_deepseek(file_with_prompts=args.batch_file,
                                   output_file=args.generated_CC_file)

        elif args.submit_to_nemotron:
            generate_with_nemotron(file_with_prompts=args.batch_file,
                                   output_file=args.generated_CC_file)

        elif args.submit_to_mixtral:
            generate_with_mixtral(file_with_prompts=args.batch_file,
                                  output_file=args.generated_CC_file)

        elif args.submit_to_qwq:
            generate_with_qwq(file_with_prompts=args.batch_file,
                              output_file=args.generated_CC_file)

        elif args.submit_to_phi:
            generate_with_phi(file_with_prompts=args.batch_file,
                              output_file=args.generated_CC_file)

        elif args.submit_to_mistral_nemo:
            generate_with_mistral_nemo(file_with_prompts=args.batch_file,
                                       output_file=args.generated_CC_file)

        elif args.submit_to_llama_3:
            generate_with_llama_3(file_with_prompts=args.batch_file,
                                  output_file=args.generated_CC_file)

        if args.postprocessing:
            if args.submit_to_openai:
                retrieve_from_openai(batch_id=args.openai_batch_id,
                                    output_file=args.generated_CC_file)
            postprocess_chains(
                original_dataset_file=args.dataset_file_jsonl,
                generation_file=args.generated_CC_file,
                output_file=args.full_dataset_with_chains
            )

    # Set analyze_chains = True for running the diagnostic evaluation A1 - A3; Section "Diagnostic Evaluation"
    # experiment_num = 1 for A1; 2 for A2; A3 is combined result of A1 and A2; run analyze_chains.py to run A4.
    if args.analyze_chains:

        if args.preprocessing:
            args.cause_column = "cause"
            args.effect_column = "effect"
            args.sub_causal_relations_file = "../../Data/Polaris/Polaris" + args.polaris + "_subrelations/Polaris" + args.polaris + "_causal_sub-relations_" + args.under_assessment +".csv"
            args.analysis_batch_file = ("../../Data/Polaris/BatchFiles_analyses/" + args.under_assessment + "/"
                                        + str(args.experiment_num) + "-Experiment/Polaris" + args.polaris + "_causal_relations_analysis_exp" + str(args.experiment_num) + "_about_" + args.under_assessment +".jsonl")
            if args.passive:
                args.analysis_batch_file = ("../../Data/Polaris/BatchFiles_analyses/" + args.under_assessment + "/"
                                            + str(
                            args.experiment_num) + "-Experiment/Polaris" + args.polaris + "_causal_relations_analysis_exp" + str(
                            args.experiment_num) + "_about_" + args.under_assessment + "_PASSIVE.jsonl")
            f = process_input_file(file_name=args.sub_causal_relations_file,
                                cause_col=args.cause_column,
                                effect_col=args.effect_column,
                                task="analyze",
                                experiment_num=args.experiment_num,
                                put_in_passive=args.passive)
            if args.submit_to_openai:
                args.analysis_batch_file = ("../../Data/Polaris/BatchFiles_analyses/" + args.under_assessment + "/"
                                            + str(
                            args.experiment_num) + "-Experiment/Polaris" + args.polaris + "_causal_relations_analysis_exp" + str(
                            args.experiment_num) + "_about_" + args.under_assessment + "_by_" + args.model_openai + ".jsonl")
                if args.passive:
                    args.analysis_batch_file = ("../../Data/Polaris/BatchFiles_analyses/" + args.under_assessment + "/"
                                                + str(
                                args.experiment_num) + "-Experiment/Polaris" + args.polaris + "_causal_relations_analysis_exp" + str(
                                args.experiment_num) + "_about_" + args.under_assessment + "_by_" + args.model_openai + "_PASSIVE.jsonl")
                create_batch_file(dictionary=f,
                                  model_name=args.model_openai,
                                  batch_file_name=args.analysis_batch_file)
            else:
                create_batch_file(dictionary=f,
                                model_name=args.under_assessment,
                                batch_file_name=args.analysis_batch_file)

        if args.submit_to_openai and not args.postprocessing:
            args.analysis_batch_file = ("../../Data/Polaris/BatchFiles_analyses/" + args.under_assessment + "/"
                                        + str(
                        args.experiment_num) + "-Experiment/Polaris" + args.polaris + "_causal_relations_analysis_exp" + str(
                        args.experiment_num) + "_about_" + args.under_assessment + "_by_" + args.model_openai + ".jsonl")
            if args.passive:
                args.analysis_batch_file = ("../../Data/Polaris/BatchFiles_analyses/" + args.under_assessment + "/"
                                            + str(
                            args.experiment_num) + "-Experiment/Polaris" + args.polaris + "_causal_relations_analysis_exp" + str(
                            args.experiment_num) + "_about_" + args.under_assessment + "_by_" + args.model_openai + "_PASSIVE.jsonl")
            upload_to_openai(file_to_be_uploaded=args.analysis_batch_file)

        else:
            batch_file, results_file = format_analysis_files(args, pol_num=args.polaris, passive=args.passive)
            upload_to_lm(model_name=args.conducting_assessment, file_with_prompts=batch_file, output_file=results_file)

        if args.postprocessing:
            retrieve_from_openai(batch_id=args.openai_batch_id,
                                 output_file=args.analysis_results_file)

if __name__ == '__main__':
    main()
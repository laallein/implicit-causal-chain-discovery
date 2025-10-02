from openai import OpenAI
import json

client = OpenAI(
    api_key=''
)

def check_if_completed(bid: str):
    batch = client.batches.retrieve(bid)
    print(batch.status)
    if batch.status == 'completed':
        return batch
    else:
        print("BATCH NOT FINALIZED")
        return ""

def retrieve_results_and_save(batch_job, result_file_name):
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content
    with open(result_file_name, 'wb') as file:
        file.write(result)

def store_in_jsonl(res,
                   output_file_name: str):
    with open(output_file_name, 'w') as f:
        if isinstance(res, list):
            for item in res:
                f.write(json.dumps(item) + '\n')
        elif isinstance(res, dict):
            f.write(json.dumps(res) + '\n')


def retrieve_from_openai(batch_id: str,
                         output_file: str):
    batch = check_if_completed(batch_id)
    if batch != "":
        retrieve_results_and_save(batch, output_file)
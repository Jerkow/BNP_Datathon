import os
from typing import List, Dict

from datathon_ai.form_company_filling import FormCompanyFilling
from datathon_ai.extractors import BasicCountryExtractor, BasicExtractor, QuestionExtractor
from datathon_ai.interfaces import FormDataModel, CountryReferential, COUNTRY_QUESTIONS_NUMBERS, \
    NOT_COUNTRY_QUESTIONS_NUMBERS

from datathon_ai.questions.utils import prepare_sentences
import time
from transformers import AutoTokenizer



dev = False

if dev:
    # Import the model
    # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
    # model = AutoModel.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
    # model.save_pretrained("./resources/distilbert-base-nli-stsb-mean-tokens")
    # tokenizer.save_pretrained("./resources/distilbert-base-nli-stsb-mean-tokens")
    # model = tokenizer

    # model = SentenceTransformer('distilroberta-base-msmarco-v2')
    # model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    model = AutoTokenizer.from_pretrained("./resources/distilbert-base-nli-stsb-mean-tokens")
else:
    model = AutoTokenizer.from_pretrained("./resources/distilbert-base-nli-stsb-mean-tokens")
    # model = SentenceTransformer('/apps/models/sentence_transformers_distilroberta_base_msmarco')


def main() -> Dict[int, int]:
    """
    USED BY DATACHALLENGE PLATFORM.
    Function that makes predictions. The .txt documents are located in the /data folder at the root of your code.
    :return: a dictionary with question_number as keys and answer_id as values.
    If number of keys in dictionary is not equaled to number_company * nb_question_by_company, it raises an
    error.
    """
    # DOCUMENTS DIRECTORY
    # Path of the directory that contains the .txt documents. One .txt document by company. IT NEEDS TO BE "/data" when you upload it in data challenge platform. For test in local, you can modifiy to match your data path.
    if dev:
        documents_directory = "../example_dataset/data"
    else: 
        documents_directory = "/data"

    path_to_files: List[str] = [os.path.join(documents_directory, file) for file in os.listdir(documents_directory)]
    assert len(path_to_files) == 10  # 10 files in documents directory
    path_to_files.sort() # Sort list of path file by alphabetical order to match ground truth annotations order : IT IS ESSENTIAL.

    # INITIALIZATION OF YOUR OBJECTS
    data_model = FormDataModel.from_json_file(
        os.path.join(os.path.dirname(__file__), "resources", "data-model.json")
    )
    country_referential = CountryReferential.from_csv(
        os.path.join(os.path.dirname(__file__), "resources", "countries_code.csv")
    )
    form_company_filling = FormCompanyFilling([
        BasicExtractor(
            question_ids=NOT_COUNTRY_QUESTIONS_NUMBERS,
            form_data_model=data_model,
        ),
        BasicCountryExtractor(
            question_ids=COUNTRY_QUESTIONS_NUMBERS,
            form_data_model=data_model,
            country_code_referential=country_referential,

        )
    ])

    # COMPUTE PREDICTION BY FILE (ie company)
    print("##################################")
    print("RUNNING PREDICTION")
    results: Dict[int, int] = {}
    start_time = time.time()
    for i, path in enumerate(path_to_files):
        start = time.time()
        print(f"File : {path}")
        with open(path, "r") as input_file:
            text = input_file.read()
        print("... Encoding ...")
        start_encoding = time.time()
        embeddings = model.encode(prepare_sentences(text))
        # embeddings = []
        print("Successfully encoded")
        print("Encoding time : ", time.time() - start_encoding)

        form_company_response = form_company_filling.fill(text, embeddings, model)
        form_company_response.sort_by_question_id() # ESSENTIAL : Sort the response by question number for each company
        for answer in form_company_response.answers:
            question_number = answer.question_id + i * 22 # ESSENTIAL : each company has 22 questions. Each question_number in results should be unique
            results[question_number] = answer.answer_id
        # gc.collect()
        print("File time :", time.time()-start, '\n')
    # CHECK FORMAT RESULTS IS DATACHALLENGE PLATFORM COMPATIBLE
    assert len(results) == len(path_to_files) * (len(COUNTRY_QUESTIONS_NUMBERS) + len(NOT_COUNTRY_QUESTIONS_NUMBERS))
    assert set(list(results.keys())) == {i for i in range(1,221)}
    print("Full Time", time.time()-start_time)
    return results


if __name__ == "__main__":
    main()


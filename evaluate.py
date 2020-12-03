import sys
from typing import Dict, List
import pandas as pd

from main import main

from datathon_ai.interfaces import NOT_COUNTRY_QUESTIONS_NUMBERS

COUNTRY_QUESTION_ID_5 = [5, 6, 7]
COUNTRY_QUESTION_ID_7 = [9, 10]
COUNTRY_QUESTION_ID_8 = [11, 12]
COUNTRY_QUESTION_ID_14 = [18, 19, 20]
COUNTRY_GROUPED_QUESTIONS = [
    COUNTRY_QUESTION_ID_5, COUNTRY_QUESTION_ID_7, COUNTRY_QUESTION_ID_8, COUNTRY_QUESTION_ID_14
]


def evalutate(annotation_path: str):
    """
    Function that computes the accuracy of form filling.
    :param annotation_path: path of the csv annotation file
    :return:

    A special computation is done for the grouped questions. A grouped question is a list of questions associated to
    the same global question. For instance questions 5, 6, 7 are associated to global questions
    "What are the mentionned recipient countries ?". For each group question, we compute them in the method
    compute_grouped_metric.
    """
    # Compute predictions and format it into a dataframe
    predictions = main()
    print("\n###############")
    print("RUNNING EVALUATION")
    predictions = [
        {"question_number": question_number, "response_id": response_id}
        for question_number, response_id in predictions.items()
    ]
    predictions_df = pd.DataFrame(predictions)
    print(f"Number of Predictions : {predictions_df.shape[0]}")
    assert set(predictions_df.columns.tolist()) == {"response_id", "question_number"}

    # Load annotation file
    ground_truth_df = pd.read_csv(annotation_path)
    print(f"Number of ground truth : {ground_truth_df.shape[0]}")

    assert ground_truth_df.shape[0] == predictions_df.shape[0]
    assert set(ground_truth_df["question_number"].unique().tolist()) == set(predictions_df["question_number"].unique().tolist())

    # Join both dataframe on keys ["question_number", "filename"]
    df_evaluate = pd.merge(
        ground_truth_df, predictions_df, on="question_number", suffixes=["_truth", "_prediction"]
    )
    assert df_evaluate.shape[0] == predictions_df.shape[0]

    # Run evaluation : accuracy of filled forms (10 documents in documents_directory)
    print("###########################")
    positive_results = 0
    res_by_questions: Dict[str, List[float]] = {}
    for i in range(0, 10):
        company_result = []
        for group in COUNTRY_GROUPED_QUESTIONS:
            group_formated = [question_number + i * 22 for question_number in group]
            score = compute_grouped_metric(df_evaluate, group_formated)
            company_result.append(score)
            group_string = "|".join([str(question_n) for question_n in group])
            if group_string in res_by_questions:
                res_by_questions[group_string].append(score)
            else:
                res_by_questions[group_string] = [score]
        for question in NOT_COUNTRY_QUESTIONS_NUMBERS:
            question_formated = question + i * 22
            df_filter_question = df_evaluate[(df_evaluate["question_number"] == question_formated)]
            assert df_filter_question.shape[0] == 1
            truth_value = df_filter_question["response_id_truth"].values[0]
            prediction = df_filter_question["response_id_prediction"].values[0]
            if truth_value == prediction:
                score = 1
            else:
                score = 0
            company_result.append(score)
            if str(question) in res_by_questions:
                res_by_questions[str(question)].append(score)
            else:
                res_by_questions[str(question)] = [score]

        company_score = sum(company_result)/len(company_result)
        positive_results += company_score
        print(f"COMPANY {i} : {company_score*100} % of form completion")

    # Results by question_id
    print("\n###########################")
    for question in res_by_questions:
        score_question_details = res_by_questions[question]
        assert len(score_question_details) == 10
        print(f"QUESTION {question} : {(sum(score_question_details)/10)*100} % of completion for all companies")

    # GLOBAL RESULT
    global_result = positive_results / 10
    print("\n###########################")
    print(f"MEAN RATIO OF FORM COMPLETION BY COMPANY : {global_result*100}")


def compute_grouped_metric(df_evaluate: pd.DataFrame, questions_in_group: List[int]) -> float:
    """
    Method that compute metric for a group question for a specific filename (ie company).
    :param df_evaluate: dataframe of evaluation merging predictions and ground truth annotations.
    Columns are : filename, question_number, response_id_truth, response_id_prediction
    :param questions_in_group: a list of questions number related to the same global question.
    :return: a score
    """
    df_filter = df_evaluate[(df_evaluate["question_number"].isin(questions_in_group))]
    assert df_filter.shape[0] == len(questions_in_group)
    # Exact match between the ground truth and prediction
    if df_filter["response_id_truth"].equals(df_filter["response_id_prediction"]):
        return 1
    # If ground truth first element is equaled to 0, then all ground truth elements in df_filter are equaled to 0.
    # It's due to annotation specification. So there is at least one element in response_id_prediction not equaled to 0.
    # So the score is 0.
    elif df_filter["response_id_truth"].iloc[0] == 0:
        return 0

    # Compare prediction and ground truth by question if first element of ground truth is not null.
    else:
        i = 0
        res = []
        for i in range(len(questions_in_group)):
            ground_truth_q = df_filter["response_id_truth"].iloc[i]
            pred_q = df_filter["response_id_prediction"].iloc[i]
            if ground_truth_q != 0:
                if ground_truth_q == pred_q:
                    res.append(1)
                else:
                    res.append(0)
            elif pred_q != 0:
                res.append(0)

        return sum(res)/len(res)


if __name__ == "__main__":
    annotation_path = sys.argv[1]
    evalutate(annotation_path)
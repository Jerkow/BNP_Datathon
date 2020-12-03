from dataclasses import dataclass
from typing import List


@dataclass
class QuestionResponse:
    """
    Interface that represents the response of one question.
    """
    answer_id: int
    question_id: int
    justification: str = None
    

@dataclass
class FormCompanyResponse:
    """
    Interface that represents the response of one form company.
    """
    answers: List[QuestionResponse]

    @classmethod
    def from_list_question_response(cls, question_responses: List[QuestionResponse]):
        data_model_questions_id = {i for i in range(1, 23)} # 22 questions in total
        extracted_questions_id = {response.question_id for response in question_responses}
        if data_model_questions_id == extracted_questions_id:
            return cls(answers=question_responses)

        raise ValueError(
            f"Missing questions number for the company : {data_model_questions_id.difference(extracted_questions_id)}"
        )
    
    def sort_by_question_id(self):
        sorted_answers = sorted(self.answers, key=lambda x: x.question_id)
        self.answers = sorted_answers
        
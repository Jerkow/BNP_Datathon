from typing import List

from datathon_ai.interfaces import QuestionResponse, FormDataModel
from datathon_ai.questions import question


class QuestionExtractor:
    """
    A QuestionExtractor is a type of extractor that extracts the answers for a list of question ids.
    """
    def __init__(self, question_ids: List[int], form_data_model: FormDataModel):
        self.question_ids = question_ids
        self.form_data_model = form_data_model
    
    def extract(self, text: str) -> List[QuestionResponse]:
        response = []
        for question_id in self.question_ids:
            question = self.form_data_model.get_specific_question_data_model(question_id)
            response.append(question(question_id, question, text))
        return response
    
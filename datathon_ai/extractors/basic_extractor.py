from typing import List

from datathon_ai.interfaces import FormDataModel, QuestionResponse
from .question_extractor import QuestionExtractor
from datathon_ai.questions import question



class BasicExtractor(QuestionExtractor):
    def __init__(self, question_ids: List[int], form_data_model: FormDataModel):
        super().__init__(question_ids, form_data_model)

    def extract(self, text: str, embeddings) -> List[QuestionResponse]:
        responses = []
        for question_id in self.question_ids:
            print(question_id)

            question_data = self.form_data_model.get_specific_question_data_model(question_id)
            responses.append(question(question_data, text, embeddings))
        return responses
    
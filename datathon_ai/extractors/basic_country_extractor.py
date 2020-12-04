from typing import List

from datathon_ai.interfaces import FormDataModel, QuestionResponse, COUNTRY_QUESTIONS_NUMBERS, CountryReferential
from .question_extractor import QuestionExtractor
from datathon_ai.questions import question



class BasicCountryExtractor(QuestionExtractor):
    def __init__(self, question_ids: List[int], form_data_model: FormDataModel,
                 country_code_referential: CountryReferential):
        for q_number in question_ids:
            assert q_number in COUNTRY_QUESTIONS_NUMBERS
        super().__init__(question_ids, form_data_model)
        self.country_code_referential = country_code_referential

    def extract(self, text: str, embeddings) -> List[QuestionResponse]:
        responses = []
        answer_dict = {}
        for question_id in self.question_ids:
            print(question_id)
            question_data = self.form_data_model.get_specific_question_data_model(question_id)
            if question_id in range(5,8):
                if question_id == 5:
                    answer_dict[5] = question(question_data, text, embeddings)
                responses.append(answer_dict[5][question_id-5] if len(answer_dict[5]) > question_id-5 else QuestionResponse(answer_id=0, question_id=question_id, justification='')) 
            elif question_id in range(9,11):
                if question_id == 9:
                    answer_dict[9] = question(question_data, text, embeddings)
                responses.append(answer_dict[9][question_id-9] if len(answer_dict[9]) > question_id-9 else QuestionResponse(answer_id=0, question_id=question_id, justification='')) 
            elif question_id in range(11, 13):
                if question_id == 11:
                    answer_dict[11] = question(question_data, text, embeddings)
                responses.append(answer_dict[11][question_id-11] if len(answer_dict[11]) > question_id-11 else QuestionResponse(answer_id=0, question_id=question_id, justification=''))
            else: 
                responses.append(question(question_data, text, embeddings))
        return responses
    
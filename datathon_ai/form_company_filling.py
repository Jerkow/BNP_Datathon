from typing import List

from datathon_ai.extractors import QuestionExtractor
from datathon_ai.interfaces import FormCompanyResponse, QuestionResponse


class FormCompanyFilling:
    def __init__(self, questions_extractors: List[QuestionExtractor]):
        self.questions_extractors = questions_extractors
    
    def fill(self, text, embeddings) -> FormCompanyResponse:
        responses = []
        for extractor in self.questions_extractors:
            extractor_response: List[QuestionResponse] = extractor.extract(text, embeddings)
            responses.extend(extractor_response)
        form_company_response = FormCompanyResponse.from_list_question_response(responses)
        return form_company_response
        
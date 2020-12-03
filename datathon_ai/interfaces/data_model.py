import json
import os
from dataclasses import dataclass
from typing import List, Dict
from marshmallow import Schema, fields, post_load


@dataclass
class QuestionAnswer:
    text_answer: str
    id_answer: int = None


class QuestionAnswerSchema(Schema):
    text_answer = fields.String()
    id_answer = fields.Integer(allow_none=True)
    
    @post_load
    def make_question_answer(self, data, **kwargs) -> QuestionAnswer:
        return QuestionAnswer(**data)


@dataclass
class QuestionDataModel:
    raw_question: str
    question_id: int
    answers: List[QuestionAnswer]


class QuestionDataModelSchema(Schema):
    raw_question = fields.String()
    question_id = fields.Integer()
    answers = fields.List(fields.Nested(QuestionAnswerSchema))
    
    @post_load
    def make_question_data_model(self, data, **kwargs) -> QuestionDataModel:
        return QuestionDataModel(**data)
    

@dataclass
class FormDataModel:
    """
    Interface that represents the data-model.json file
    """
    questions: List[QuestionDataModel]

    @classmethod
    def from_json_file(cls, path_to_file: str) -> 'FormDataModel':
        assert os.path.exists(path_to_file)
        with open(path_to_file, "r") as input_file:
            data_model_as_dict: Dict = json.load(input_file)

        return FormDataModelSchema().load(data_model_as_dict)

    def get_specific_question_data_model(self, question_number: int) -> QuestionDataModel:
        question = [question for question in self.questions if question.question_id == question_number]
        assert len(question) == 1
        return question[0]


class FormDataModelSchema(Schema):
    questions = fields.List(fields.Nested(QuestionDataModelSchema))
    
    @post_load
    def make_form_data_model(self, data, **kwargs) -> FormDataModel:
        return FormDataModel(**data)
    
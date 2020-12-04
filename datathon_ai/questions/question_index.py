from .question5_to_12 import question5, question8, question9, question11
from .questions13_14_21_22 import is_ISO_27001_certified, is_cost_mentioned, is_audit_right_mentioned, is_retention_date_mentioned
from datathon_ai.interfaces import FormDataModel, QuestionResponse


def question(question, text, embeddings):
    question_id = question.question_id
    if question_id == 5:
        return question5(question, text, embeddings)
    elif question_id == 8:
        return question8(question, text)
    elif question_id == 9:
        return question9(question, text, embeddings)
    elif question_id == 11:
        return question11(question, text, embeddings)
    elif question_id == 13:
        return is_ISO_27001_certified(text)
    elif question_id == 14:
        return is_cost_mentioned(text, embeddings)
    elif question_id == 21:
        return is_audit_right_mentioned(text, embeddings)
    elif question_id == 22:
        return is_retention_date_mentioned(text)
    else:
        return(QuestionResponse(answer_id=1, question_id=question_id, justification='justification'))

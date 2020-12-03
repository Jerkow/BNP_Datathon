from .question5_to_12 import question5, question8, question9, question11
from datathon_ai.interfaces import FormDataModel, QuestionResponse



def question(question, text):
    question_id = question.question_id
    if question_id == 5:
        return question5(question,text)
    elif question_id == 8:
        return question8(question, text)
    elif question_id == 9:
        return question9(question, text)
    elif question_id == 11:
        return question11(question, text)
    else: return(QuestionResponse(answer_id=0, question_id=question_id, justification='justification'))



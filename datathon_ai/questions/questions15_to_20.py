from datathon_ai.interfaces import QuestionResponse


def question15_16(text):
    text_to_split = text
    splitted = text_to_split.split("\n")

    share_paragraphs = []
    for paragraph in splitted:
        paragraph_lower = paragraph.lower()
        if ("transfer" in paragraph_lower or "share" in paragraph_lower) and ("parties" in paragraph_lower or "providers" in paragraph_lower):
            share_paragraphs.append(paragraph_lower)
    if len(share_paragraphs) > 0:
        for para in share_paragraphs:
            if "purposes" in para:
                return [QuestionResponse(answer_id=1, question_id=15, justification=para), QuestionResponse(answer_id=1, question_id=16, justification=para)]

        return [QuestionResponse(answer_id=1, question_id=15, justification=share_paragraphs), QuestionResponse(answer_id=0, question_id=16, justification="")]
    return [QuestionResponse(answer_id=0, question_id=15, justification=""), QuestionResponse(answer_id=0, question_id=16, justification="")]

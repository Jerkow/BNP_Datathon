# import nltk
# nltk.download('punkt')

from nltk.tokenize import word_tokenize
from scipy import spatial
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from datathon_ai.interfaces import FormDataModel, QuestionResponse


# model = AutoTokenizer.from_pretrained('./resources/distilbert-base-nli-stsb-mean-tokens/')
# model = SentenceTransformer('distilroberta-base-msmarco-v2')
model = SentenceTransformer('/apps/models/sentence_transformers_distilroberta_base_msmarco')



def is_ISO_27001_certified(whole_text):
    whole_text_lower = whole_text.lower()
    return QuestionResponse(answer_id=1 if ("iso27001" in whole_text_lower) or ("iso 27001" in whole_text_lower) else 0, question_id=13, justification='')


def is_cost_mentioned(whole_text, embeddings):

    keywords = ['payment', 'fee', 'pric', 'subscri',
                'plan', 'bill', 'purchas', 'licens']
    # 'cost'
    questions = ['Is there a payment needed?', 'Are there fees for use?', 'What is the price of the product or service?',
                 'Are there any subscription plans?', 'Are there any billings?', 'Is there any license needed to use the product or service?']
    # 'What is the cost of the product or service?'
    question_embeddings = model.encode(questions)
    paragraphs = [paragraph for paragraph in whole_text.lower().split(
        "\n") if len(paragraph) > 0]
    scores = [0 for _ in paragraphs]
    thresh = 0.45   # normalement à trouver par reg log à partir de la seule feature "mesure de similarité" ?

    for i in range(len(paragraphs)):

        paragraph = paragraphs[i]
        tokens = word_tokenize(paragraph)
        score = 0

        if len(tokens) < 5:
            continue

        # paragraph_embedding = model.encode(paragraph)
        paragraph_embedding = embeddings[i]

        for token in tokens:

            for keyword in keywords:

                if token[:len(keyword)].lower() == keyword:

                    for question_embedding in question_embeddings:

                        score += spatial.distance.cosine(
                            paragraph_embedding, question_embedding)

                    score /= len(questions)
                    scores[i] = score

                    break
    return QuestionResponse(answer_id=1 if max(scores) > thresh else 0, question_id=14, justification='')


def is_audit_right_mentioned(whole_text, embeddings):

    keywords = ['audit right', 'right to audit',
                'audit rights', 'right of audit', 'full rights']
    root = 'audit'
    questions = ['Are there full rights for the user in the GDPR?', 'Will the user have audit rights?',
                 'Does the user have the right to audit his data?', 'Is auditing possible?']
    question_embeddings = model.encode(questions)
    paragraphs = [paragraph for paragraph in whole_text.lower().split(
        "\n") if len(paragraph) > 0]
    scores = [0 for _ in paragraphs]
    thresh = 0.4    # normalement à trouver par reg log à partir de la seule feature "mesure de similarité" ?

    for i in range(len(paragraphs)):

        paragraph = paragraphs[i]
        tokens = word_tokenize(paragraph)
        score_root = 0
        score_keywords = 0

        if "28" in paragraph:
            return QuestionResponse(answer_id=1, question_id=21, justification='')   # on sait direct que c'est bon

        if len(tokens) < 5:
            continue

        # paragraph_embedding = model.encode(paragraph)
        paragraph_embedding = embeddings[i]
        for token in tokens:

            if token[:len(root)].lower() == root:

                for question_embedding in question_embeddings:

                    score_root += spatial.distance.cosine(
                        paragraph_embedding, question_embedding)
                    score_root /= len(questions)
                    break

        for keyword in keywords:

            if keyword in paragraph.lower():

                score_keywords_para = []

                for question_embedding in question_embeddings:

                    score_keywords_para.append(spatial.distance.cosine(
                        paragraph_embedding, question_embedding))

                score_keywords = max(score_keywords_para)
                break

        scores[i] = max(score_keywords, score_root)

    return QuestionResponse(answer_id=1 if max(scores) > thresh else 0, question_id=21, justification='')


def is_retention_date_mentioned(whole_text):
    # pas encore implémenté
    whole_text_lower = whole_text.lower()
    return QuestionResponse(answer_id=1 if ("retention date" in whole_text_lower) else 0, question_id=22, justification='')


# définir les fonctions questions ?

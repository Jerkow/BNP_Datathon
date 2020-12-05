import numpy as np
from nltk.tokenize import word_tokenize
from scipy import spatial
from datathon_ai.interfaces import QuestionResponse
from .utils import prepare_sentences

import spacy
nlp = spacy.load("/apps/models/ner_spacy_en")


def is_ISO_27001_certified(whole_text):
    
    whole_text_lower = whole_text.lower()
    paragraphs = whole_text_lower.split("\n")
    for paragraph in paragraphs:
        if ("iso27001" in paragraph) or ("iso 27001" in paragraph):
            return QuestionResponse(answer_id=1, question_id=13, justification=paragraph)
    return QuestionResponse(answer_id=0, question_id=13, justification=None)


def is_cost_mentioned(whole_text, embeddings, model):
    
    keywords = ['payment', 'fee', 'pric', 'subscri', 'plan', 'bill', 'purchas', 'licens']
    # 'cost'
    questions = ['Is there a payment needed?', 'Are there fees for use?', 'What is the price of the product or service?', 'Are there any subscription plans?', 'Are there any billings?', 'Is there any license needed to use the product or service?']
    # 'What is the cost of the product or service?'
    question_embeddings = model.encode(questions)
    paragraphs = [paragraph for paragraph in whole_text.lower().split("\n") if len(paragraph) > 0]
    scores = [(0, paragraph) for paragraph in paragraphs]
    thresh = 0.45   # normalement à trouver par reg log à partir de la seule feature "mesure de similarité" ?
    
    for i in range(len(paragraphs)):
        
        paragraph = paragraphs[i]
        tokens = word_tokenize(paragraph)
        score = 0
        
        if len(tokens) < 5:
            continue
            
        paragraph_embedding = embeddings[i]
            
        for token in tokens:
            
            for keyword in keywords:
                
                if token[:len(keyword)].lower() == keyword:
                    
                    for question_embedding in question_embeddings:
                        
                        score += spatial.distance.cosine(paragraph_embedding, question_embedding)
                    
                    score /= len(questions)
                    scores[i][0] = score
                    
                    break

    max_index = np.argmax([score[0] for score in scores])
    return QuestionResponse(1, 14, scores[max_index][1]) if scores[max_index][0] > thresh else QuestionResponse(0, 14, None)


def is_audit_right_mentioned(whole_text, embeddings, model):
    
    keywords = ['audit right', 'right to audit', 'audit rights', 'right of audit', 'full rights']
    root = 'audit'
    questions = ['Are there full rights for the user in the GDPR?', 'Will the user have audit rights?', 'Does the user have the right to audit his data?', 'Is auditing possible?']
    question_embeddings = model.encode(questions)
    paragraphs = [paragraph for paragraph in whole_text.lower().split("\n") if len(paragraph) > 0]
    scores = [(0, paragraph) for paragraph in paragraphs]
    thresh = 0.4    # normalement à trouver par reg log à partir de la seule feature "mesure de similarité" ?
    
    for i in range(len(paragraphs)):
        
        paragraph = paragraphs[i]
        tokens = word_tokenize(paragraph)
        score_root = 0
        score_keywords = 0
        
        if "28" in paragraph:
            return QuestionResponse(1, 21, 'article 28 mentioned')
            # on sait direct que c'est bon
        
        if len(tokens) < 5:
            continue
        
        paragraph_embedding = embeddings[i]
        
        for token in tokens:

            if token[:len(root)].lower() == root:

                for question_embedding in question_embeddings:

                    score_root += spatial.distance.cosine(paragraph_embedding, question_embedding)
                    score_root /= len(questions)
                    break                         
    
        for keyword in keywords:

            if keyword in paragraph.lower():
                
                score_keywords_para = []

                for question_embedding in question_embeddings:

                    score_keywords_para.append(spatial.distance.cosine(paragraph_embedding, question_embedding))

                score_keywords = max(score_keywords_para)
                break
                
        scores[i][0] = max(score_keywords, score_root)
        
    max_index = np.argmax([score[0] for score in scores])
    return QuestionResponse(1, 21, scores[max_index][1]) if scores[max_index][0] > thresh else QuestionResponse(0, 21, None)


def is_retention_date_mentioned(whole_text):
    # pas encore bien implémenté
    whole_text_lower = whole_text.lower()
    paragraphs = whole_text_lower.split("\n")
    for paragraph in paragraphs:
        condition_ret = "retention" in paragraph or any(["stor" in word[:min(len("stor"),len(word))-1] for word in paragraph.split()])
        condition_date = "date" in paragraph or "duration" in paragraph or "period" in paragraph
        if (
            (condition_ret and condition_date)
            or ("article 5" in paragraph)
            or ("art. 5" in paragraph)
            or ("art5" in paragraph)
        ):
            tokens = nlp(paragraph)
            for ent in tokens.ents:
                if ent.label_ == "DATE":
                    return QuestionResponse(2, 22, ent.text)
            return QuestionResponse(1, 22, paragraph)
    return QuestionResponse(0, 22, None)

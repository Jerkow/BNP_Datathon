from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
import pandas as pd
import numpy as np
from datathon_ai.interfaces import QuestionResponse
from .utils import countries_dict, names, eu, demonyms


# model = SentenceTransformer('distilroberta-base-msmarco-v2')
# model = SentenceTransformer(
#     '/apps/models/sentence_transformers_distilroberta_base_msmarco')


questions = {5: {"question": "In which countries outside of the EU the data can be transferred to ?", "key_words": ["transfer"]},
             9: {"question": "What is the country of the applicable law of the contract?", "key_words": ["applicable law", "applicable", "applicable laws"]},
             11: {"question": "What is the country jurisdiction applicable in the event of a dispute?", "key_words": ["jurisdiction"]}}


def cosine(u, v):
    return np.abs(np.dot(u, v)) / np.sqrt(np.linalg.norm(u) * np.linalg.norm(v))


def moy_gliss(liste, add, n):
    liste.append(add)
    if len(liste) > n:
        liste = liste[1:]
    moy_sim = sum(liste)/len(liste)
    return liste, moy_sim


def corresponds_to(country_list, identifier):
    return identifier.replace('.', '').lower() in country_list


def getId(name, countries_dict):
    for i in countries_dict.keys():
        country = countries_dict[i]
        if corresponds_to(country, name):
            return i
    return -1


def prepare_sentences(sentences):
    text_list = sentences.split("\n")
    text_list = [a for a in text_list if a != '']
    return text_list


# def get_country_data():
#     countries_key = 'resources/countries_code.csv'
#     eu_key = 'resources/eu.csv'

#     # Load data into a Pandas Data Frame

#     countries = pd.read_csv(countries_key)
#     eu = pd.read_csv(eu_key)
#     eu = list(np.array(eu.values).transpose()[0])
#     names = countries["name"]
#     names = list(names) + ["U.S.", "U.K."]
#     countries_dict = {}
#     for country in countries.values:
#         countries_dict[country[0]] = list(country[1:])

#     countries_demonym_key = 'resources/countries_demonym.csv'
#     countries_demonym = pd.read_csv(countries_demonym_key)
#     demonyms = list(countries_demonym["fdemonym"]) + \
#         list(countries_demonym["mdemonym"])

#     for i in countries_dict.keys():
#         country = countries_demonym[countries_demonym["id"] == i]
#         countries_dict[i] += list(country["fdemonym"]) + \
#             list(country["mdemonym"])
#         countries_dict[i] = [str(c).lower() for c in countries_dict[i]]
#     return countries_dict, names, eu, demonyms


# countries_dict, names, eu, demonyms = get_country_data()



def get_paragraph(question, sentences, embeddings, model):
    # Retrieve question Data
    question_id = question.question_id
    question = question.raw_question
    key_words = questions[question_id]["key_words"]
    question_vec = model.encode([question])[0]
    sentences = prepare_sentences(sentences)
    # First find the paraphraphs

    similarities = [0]*len(sentences)
    max_sim = 0
    index = 0
    n = 10
    for i in range(len(sentences)):
        count = 0
        for word in key_words:
            count += sentences[i].count(word)
        if count > 0:
            sim = 0
            for k in range(-n, n+1):
                sim += cosine(question_vec, embeddings[i])
            similarities[i] = sim*count
            if sim > max_sim:
                max_sim = sim
                index = i

    # Do the average to find the best paragraph
    moy_list = []
    moy_similarities = [0]*len(sentences)

    for i in range(len(similarities)):
        moy_list, moy = moy_gliss(moy_list, similarities[i], n)
        moy_similarities[i] = moy

    k = np.argmax(moy_similarities)
    paragraph = sentences[int(k): int(k+2*n)]
    return paragraph

# Question 5, 6, 7


def question5(question, sentences, embeddings, model):
    # Retrieve Data
    paragraph = get_paragraph(question, sentences, embeddings, model)
    # Initialize Spacy
    nlp = English()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(name) for name in names]
    matcher.add("Names", None, *patterns)
    responseNames = set([])
    # Get the matches
    for s in paragraph:
        doc = nlp(s)
        matches = matcher(doc)
        for m in matches:
            country = str(doc[m[1]:m[2]])
            if country not in eu:
                responseNames.add(country)
    response = set([getId(name, countries_dict) for name in responseNames])
    response = sorted(list(response))
    return [QuestionResponse(answer_id=response[i], question_id=5+i, justification=paragraph) for i in range(len(response))]

# Question 8


def question8(question, sentences):
    
    sentences_lower = sentences.lower()
    for word in ["Binding Corporate Rules", "BCR", "Standard Contractual Clauses", "SCC"]:
        index = sentences_lower.find(word.lower())
        if index >= 0:
            return QuestionResponse(answer_id = 1, question_id = 8, justification = sentences[index: index + 100])
    return QuestionResponse(answer_id = 0, question_id = 8, justification = '')
    


# Questions 9, 10


def question9(question, sentences, embeddings, model):
    # Retrieve Data
    paragraph = get_paragraph(question, sentences, embeddings, model)
    # Initialize Spacy
    nlp = English()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(str(name)) for name in demonyms+names]
    matcher.add("Names", None, *patterns)
    responseNames = set([])

    # Get the matches
    for s in paragraph:
        doc = nlp(s)
        matches = matcher(doc)
        for m in matches:
            country = str(doc[m[1]:m[2]])
            responseNames.add(country)
    response = set([getId(name, countries_dict) for name in responseNames])
    response = sorted(list(response))
    return [QuestionResponse(answer_id=response[i], question_id=9+i, justification=paragraph) for i in range(len(response))]


def question11(question, sentences, embeddings, model):
    # Retrieve Data
    paragraph = get_paragraph(question, sentences, embeddings, model)
    # Initialize Spacy
    nlp = English()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(str(name)) for name in demonyms+names]
    matcher.add("Names", None, *patterns)
    responseNames = set([])

    # Get the matches
    for s in paragraph:
        doc = nlp(s)
        matches = matcher(doc)
        for m in matches:
            country = str(doc[m[1]:m[2]])
            responseNames.add(country)
    response = set([getId(name, countries_dict) for name in responseNames])
    response = sorted(list(response))
    return [QuestionResponse(answer_id=response[i], question_id=11+i, justification=paragraph) for i in range(len(response))]

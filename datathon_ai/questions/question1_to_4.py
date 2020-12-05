#imports
import spacy
from datathon_ai.interfaces import FormDataModel, QuestionResponse
nlp = spacy.load("/apps/models/ner_spacy_en")
# nlp = spacy.load("en_core_web_sm")

from .utils import eu


def question1(text):
    text_lower = text.lower()
    key_words = ["Personal data", "Personal information"]
    for key in key_words:
        index = text_lower.find(key.lower())
        if index != -1:
            justification = text[index : index + len(key) + 200]
            return QuestionResponse(answer_id=1, question_id=1, justification=justification)
    return QuestionResponse(answer_id=0, question_id=1, justification='')


def question2(text):
    text_lower = text.lower()
    key_words = ["DPA", "Data processing agreement", "Data protection agreement", "Data processing addendum", "Data protection addendum"]
    for key in key_words:
        index = text_lower.find(key.lower())
        if index != -1:
            justification = text[index : index + len(key) + 200]
            return QuestionResponse(answer_id=1, question_id=2, justification=justification)
    return QuestionResponse(answer_id=0, question_id=2, justification='')

def question3_4(text):
    text_to_split = text
    splitted = text_to_split.split("\n")

    transfer_data_info_paragraphs = []
    for paragraph in splitted:
        paragraph_lower = paragraph.lower()
        if "transfer" in paragraph_lower and "data" in paragraph_lower:
            transfer_data_info_paragraphs.append(paragraph)

        
    transfer_countries = []
    transfer_countries_paragraph = []
    
    #seeing countries if mentionned
    for prgph in transfer_data_info_paragraphs:
        counter=0
        prgph_doc = nlp(prgph)
        for ent in prgph_doc.ents:
            if ent.label_=="GPE":
                counter+=1
                transfer_countries.append(ent.text)
        if counter > 0:
            transfer_countries_paragraph.append(prgph)

    if transfer_countries_paragraph != []:
        countries_out_europe = []
        for localisation in transfer_countries:
            is_in_europe = localisation.lower() in eu
            continent = 'EU' if is_in_europe else ' '
            if continent != 'EU':
                countries_out_europe.append(localisation)
                transfer_out_of_europe_paragraph = []
                for par in transfer_countries_paragraph:
                    if localisation in par:
                        transfer_out_of_europe_paragraph.append(par)

        if countries_out_europe != []:
            key_words = ["Other countries", "Other country", "Third countries", "Third country", "Country out of", "Countries out of", "Country outside", "countries outside"]
            counter = 0
            for parg in transfer_out_of_europe_paragraph:
                parg_lower = parg.lower()
                for key in key_words:
                    if key.lower() in parg_lower:
                        return [QuestionResponse(answer_id=1, question_id=3, justification=""), QuestionResponse(answer_id=0, question_id=4, justification="")]
                    else:
                        counter +=1
            if counter > 0:
                return [QuestionResponse(answer_id=1, question_id=3, justification=""), QuestionResponse(answer_id=1, question_id=4, justification="")]
        
        else:
            return [QuestionResponse(answer_id=1, question_id=3, justification=""), QuestionResponse(answer_id=0, question_id=4, justification="")]
    else:
            return [QuestionResponse(answer_id=0, question_id=3, justification=""), QuestionResponse(answer_id=0, question_id=4, justification="")]           
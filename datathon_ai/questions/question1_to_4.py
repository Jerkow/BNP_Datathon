#imports
import spacy
from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
from datathon_ai.interfaces import FormDataModel, QuestionResponse
import pycountry
import pycountry_convert as pc
nlp = spacy.load("en_core_web_sm")


def question1(text):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(name) for name in ["Personal data", "Personal information"]]
    matcher.add("Names", None, *patterns)
    doc = nlp(text)
    matches = matcher(doc)
    answers = [0, 1]
    justification = ''
    if matches != []:
        for m in matches:
            justification += ' '.join([str(a) for a in list(doc[m[0]: m[1] + 20])])
    return QuestionResponse(answer_id=answers[matches != []], question_id=1, justification=justification)


def question2(text):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(name) for name in ["DPA", "Data processing agreement", "Data protection agreement", "Data processing addendum", "Data protection addendum"]]
    matcher.add("Names", None, *patterns)
    doc = nlp(text)
    matches = matcher(doc)
    answers = [0, 1]
    justification = ''
    if matches != []:
        for m in matches:
            justification += ' '.join([str(a) for a in list(doc[m[0]: m[1] + 20])])
    return QuestionResponse(answer_id=answers[matches != []], question_id=2, justification=justification)


def question3_4(text):
    text_to_split = text
    splitted = text_to_split.split("\n")

    # Add match ID "Transfer..." with no callback and one pattern
    matcher = Matcher(nlp.vocab)
    pattern1 = [{"LOWER": "transfer"} ]
    pattern2 = [{"LOWER": "transferred"}]
    pattern3 = [{"LOWER": "transfers"}]
    matcher.add("Transfer", None, pattern1)
    matcher.add("Transferred", None, pattern2)
    matcher.add("Transfers", None, pattern3)

    #extracting paragraphs with word "transfer" in it
    transfer_paragraphs = []
    for paragraph in splitted:
        doc = nlp(paragraph)
        matches = matcher(doc)
        if matches != []:
            transfer_paragraphs.append(paragraph)

    transfer_data_info_paragraphs = []

    if transfer_paragraphs != []:
        # Add match ID "Data" and "Information" with no callback and one pattern
        matcher = Matcher(nlp.vocab)
        pattern1 = [{"LOWER": "data"}]
        #pattern2 = [{"LOWER": "information"}]
        matcher.add("Data", None, pattern1)
        #matcher.add("Information", None, pattern2)

        #extracting paragraphs with word "data"/"information" from those with word "transfer"
        for paragraph in transfer_paragraphs:
            doc = nlp(paragraph)
            matches = matcher(doc)
            if matches != []:
                transfer_data_info_paragraphs.append(paragraph)
            
        transfer_countries = []
        transfer_countries_paragraph = []
    
        if transfer_data_info_paragraphs != []:
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
                    try:
                        countries = pycountry.countries.search_fuzzy(localisation)
                        if len(countries) == 1:
                            country = countries[0]
                            continent = pc.country_alpha2_to_continent_code(country.alpha_2)
                            if continent != 'EU':
                                countries_out_europe.append(localisation)
                                transfer_out_of_europe_paragraph = []
                                for par in transfer_countries_paragraph:
                                    if localisation in par:
                                        transfer_out_of_europe_paragraph.append(par)
                    except:
                        continue
            
                if countries_out_europe != []:
                    last_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
                    patterns = [nlp.make_doc(name) for name in ["Other countries", "Other country", "Third countries", "Third country", "Country out of", "Countries out of", "Country outside", "country outside"]]
                    last_matcher.add("Names", None, *patterns)
                    counter = 0
                    for parg in transfer_out_of_europe_paragraph:
                        last_doc = nlp(parg)
                        if(last_matcher(last_doc)!=[]):
                            print(1, " Yes&No")
                            counter = 0
                            return [QuestionResponse(answer_id=1, question_id=3, justification=""), QuestionResponse(answer_id=0, question_id=4, justification="")]
                        else:
                            counter +=1
                    if counter > 0:
                        return [QuestionResponse(answer_id=1, question_id=3, justification=""), QuestionResponse(answer_id=1, question_id=4, justification="")]
                
                else:
                    return [QuestionResponse(answer_id=1, question_id=3, justification=""), QuestionResponse(answer_id=0, question_id=4, justification="")]
            else:
                return [QuestionResponse(answer_id=1, question_id=3, justification=""), QuestionResponse(answer_id=0, question_id=4, justification="")]
        else:
            return [QuestionResponse(answer_id=0, question_id=3, justification=""), QuestionResponse(answer_id=0, question_id=4, justification="")]
                
    else:
        return [QuestionResponse(answer_id=0, question_id=3, justification=""), QuestionResponse(answer_id=0, question_id=4, justification="")]

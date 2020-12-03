# datathon-ai
Template of code for AI, NLP, Banking datathon co-organized by BNP Paribas, Paris Digital Lab of CentraleSupelec
and ILLUIN Technology.

## Contents
1. Installation
2. How to submit your code to datathon platform ?
3. To make prediction with justification, needed by Bivwak for justification verification.
4. To run evaluation in local

## Installation

### Prerequisites
* python 3.7.3

### Package installation
The packages enables by datathon platform are specified in the file : resources/platform_requirements.txt. You cannot
use a package that is not included in this file.
```bash
pip install -r requirements.txt
```

## How to submit your code to datathon platform ?
### Prerequisites
- [IMPORTANT] For more details about the specification of the main.py script, check the document : `Guide to submit a response - Datathon BNPP - EN
- You will upload your prediction code in the datathon platform. There is not internet access in this platform. So you cannot have code that depends on the connection (model loading from internet, api package, etc.)
- The maximum size of the code is 900 MB : do not upload useless items such as venv folder.
- 5 pre-trained models are available at these locations in the datathon platform environment :
  - /apps/models/bert_base_ner : Details about it https://huggingface.co/dslim/bert-base-NER
  - /apps/models/t5_boolean_questions_ramsrigouthamg : Details about it https://huggingface.co/ramsrigouthamg/t5_boolean_questions
  - /apps/models/distilbert_base_cased_distilled_squad : Details about it https://huggingface.co/distilbert-base-cased-distilled-squad
  - /apps/models/sentence_transformers_distilroberta_base_msmarco : Details about it https://www.sbert.net/docs/pretrained-models/msmarco-v2.html
  - /apps/models/ner_spacy_en : Details about it https://spacy.io/usage/linguistic-features

If you use it, no need to upload these models in the platform (keep space in your code for your own trained model).

### Steps
- Go to the data challenge platform
- Zip your code : the main.py has to be at the root of the zip.
- Upload the zip.

To run the template main script in LOCAL : change the variable `documents_directory` to match the path of the directory that contains the 10 .txt files.
WARNING : When you upload your code to the DATA CHALLENGE PLATFORM the variable `documents_directory` MUST BE EQUALED TO "/data". 
```bash
python main.py 
```

## To make prediction with justification, needed by Bivwak for justification verification.
To dump prediction for companies, the script main_with_justification.py is available.

To run the script :
```bash
python main_with_justification.py <documents_directory> <output_directory>
```

The .txt dumped file has the following structure.
### Company : docusign.txt ###
Question 1 | Answer 0 | Justification DocuSign, Inc. and its group of companies (“us,” “our,” or “we”) collect and use personal information from customers and other individuals (collectively "you").


## To run evaluation in local
To run local evaluation of your prediction, the script evaluate.py is available.

To run the evaluation script :
```bash
python evaluate.py <annotation_csv_file>
```

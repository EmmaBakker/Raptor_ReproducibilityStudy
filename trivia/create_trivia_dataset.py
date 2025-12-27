# eval_val: dict_keys(['paper_id', 'question_id', 'doc_text', 'question', 'gold_answers'])

import json, argparse, os
from pathlib import Path
from typing import List, Dict

QUESTION_VERIFIED_EVAL = "QuestionPartOfVerifiedEval"
DOC_VERIFIED_EVAL = "DocPartOfVerifiedEval"
ENTITY_PAGES = "EntityPages"
CATEGORY = "wikipedia"

def load_qa(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r") as f:
        json_data = json.load(f)
        all_data = json_data["Data"]

        for entry in all_data:
            if QUESTION_VERIFIED_EVAL not in entry.keys() or entry[QUESTION_VERIFIED_EVAL] == False:
                continue
            if ENTITY_PAGES not in entry.keys():
                continue

            entity_pages = entry["EntityPages"]
            verified_entity_pages = []
            for page in entity_pages:
                if DOC_VERIFIED_EVAL not in page.keys() or page[DOC_VERIFIED_EVAL] == False:
                    continue
                verified_entity_pages.append(page)
            
            if len(verified_entity_pages) == 0:
                continue
            data.append(entry)
    
    return data
            
def load_document_by_category_and_name(category: str, name: str) -> str:
    path = Path("./evidence") / category / name

    with open(path, "r") as f:
        doc_text = f.read()

    return doc_text

def extract_answers(question: Dict) -> List[str]:

    if 'Answer' not in question.keys():
        return None

    answers = question['Answer']

    answers_set = set()
    if 'Aliases' in answers.keys():
        answers_set.update(answers['Aliases'])

    if 'HumanAnswers' in answers.keys():
        answers_set.update(answers['HumanAnswers'])

    if 'NormalizedAliases' in answers.keys():
        answers_set.update(answers['NormalizedAliases'])

    return list(answers_set)

def get_document_filenames(entity_pages: Dict) -> List[str]:
    filenames = set()

    for page in entity_pages:
        file_name = page["Filename"]
        filenames.add(file_name)
    return list(filenames)

def create_dataset(category: str, path: Path):

    qa_list = load_qa(path)

    print("Loaded qa...")

    dataset: List[Dict] = []

    for qa in qa_list:
        answers = extract_answers(qa)
        question = qa['Question']
        question_id = qa['QuestionId']

        if len(answers) == 0:
            continue
        
        document_names = get_document_filenames(qa[ENTITY_PAGES])

        if len(document_names) == 0:
            continue

        doc_text = ""
        doc_id = ""
        for document_name in document_names:
            doc_id += document_name
            doc_text += load_document_by_category_and_name(category, document_name)

        sample = {"doc_id": doc_id, 
                "question_id": question_id, 
                "doc_text": doc_text, 
                "question": question, 
                "answers": answers}
        dataset.append(sample)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    path_wikipedia = Path("./qa/verified-wikipedia-dev.json")
    output_path = Path("./eval_val.jsonl")
    dataset = create_dataset(CATEGORY, path_wikipedia)

    with open(output_path, "w") as f:
        for sample in dataset:
            f.write(json.dumps(sample)+"\n")
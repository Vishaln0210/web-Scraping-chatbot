import spacy
from transformers import pipeline
import requests
from bs4 import BeautifulSoup


nlp = spacy.load("en_core_web_sm")


q_pipe = pipeline("question-answering")

def fetch_website_text(url):
    """
    Fetches text content from a given URL using web scraping.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad response status
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from all paragraphs on the webpage
        paragraphs = soup.find_all('p')
        text = '\n'.join([p.get_text() for p in paragraphs])
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from URL: {e}")
        return None

def answer_question(question, context):
    """
    Answers a question based on the provided context using the question-answering pipeline.
    """
    result = q_pipe(question=question, context=context)
    return result['answer']

def main():
    url = "https://en.wikipedia.org/wiki/India"
    website_text = fetch_website_text(url)

    if website_text:
   
        doc = nlp(website_text)

        relations = []

 
        for sent in doc.sents:
            for ent in sent.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "NORP"]:
                    for token in sent:
                        if token.dep_ in ["attr", "nsubj", "dobj", "pobj", "conj", "appos", "acl", "relcl"]:
                            relations.append((ent.text, token.text, sent.text))


        print("Relations in Website Text:")
        for name, rela, context in relations:
            print(f"Relation: {name} -> {rela}")
            print(f"Explanation: '{name}' is related to '{rela}' in the context of '{context}'\n")

        while True:
            question = input("Enter a Question (or 'exit' to quit): ").strip()

            if question.lower() == 'exit':
                print("Goodbye!")
                break

            answer = answer_question(question, website_text)
            if answer:
                print(f"Answer: {answer}\n")
            else:
                print("I'm sorry, I'm not sure how to answer that.\n")

    else:
        print("Unable to fetch website content. Exiting.")

if _name_ == "_main_":
    main()

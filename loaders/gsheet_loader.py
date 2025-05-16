import pandas as pd
from langchain.schema import Document

def load_google_sheet(csv_url: str):
    """
    Downloads data from Google Sheet (exported as CSV) and converts each row into a Document
    Expects columns: Question, How to Respond, Goal, Do Not
    """

    df = pd.read_csv(csv_url)

    docs = []

    for _, row in df.iterrows():
        question = str(row.get("Question", "")).strip()
        response = str(row.get("How to Respond", "")).strip()
        goal = str(row.get("Goal", "")).strip()
        do_not = str(row.get("Do Not", "")).strip()

        if not question and not response:
            continue  #Skip empty rows

        content = f"""Question: {question}
How to Respond: {response}
Goal: {goal}
Do Not: {do_not}"""

        doc = Document(
            page_content=content,
            metadata={
                "source": "google_sheet",
                "goal": goal,
                "original_question": question
            }
        )

        docs.append(doc)

    print(f"✅ Загружено строк из Google Sheet: {len(docs)}")
    return docs
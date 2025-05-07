import json
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage
load_dotenv()

def get_recommender_runner(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2
) -> RunnableSequence:
    prompt = PromptTemplate(
        input_variables=["fundamentals_json", "sentiments_json", "user_preferences", "scoring_method"],
        template="""
            You are a financial advisor. Based on the following fundamentals, sentiments, user preferences, and scoring instructions:

            1) FUNDAMENTALS as JSON:
            {fundamentals_json}

            2) NEWS SENTIMENTS as JSON:
            {sentiments_json}

            3) User preferences as JSON:
            {user_preferences}

            4) Scoring method description:
            {scoring_method}

            Rank the top 5 tickers for this user from most to least suitable using the provided scoring method, and for each ticker give a 2-sentence rationale.
            Output in JSON array form (pure JSON, no markdown fences):
            [
                {{
                    "ticker": "AAPL",
                    "score": 0.87,
                    "rationale": "..."
                }},
                ...
            ]
        """.strip()
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=api_key
    )
    return RunnableSequence(prompt, llm)


def run_recommender(
    fundamentals: dict,
    sentiments: dict,
    user_preferences: dict,
    scoring_method: str = "sentiment_score * 0.5 + YoY_revenue_growth * 0.3 + net_profit_margin * 0.2",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2
) -> list:
    runner = get_recommender_runner(model_name=model_name, temperature=temperature)
    inputs = {
        "fundamentals_json": json.dumps(fundamentals),
        "sentiments_json":  json.dumps(sentiments),
        "user_preferences": json.dumps(user_preferences),
        "scoring_method":   scoring_method
    }
    raw_output = runner.invoke(inputs)

    # Extract the text
    if isinstance(raw_output, AIMessage):
        text = raw_output.content
    else:
        text = str(raw_output)
    text = text.strip()

    # Strip out any markdown fences
    m = re.search(r"```json\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    else:
        m2 = re.search(r"```(.*?)```", text, re.S)
        if m2:
            text = m2.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("Failed to parse JSON from LLM output:")
        print(text)
        raise


if __name__ == "__main__":
    # locate src/data relative to this file
    src_dir = Path(__file__).resolve().parent.parent
    data_dir = src_dir / "data"

    def load_json(fn):
        return json.loads((data_dir / fn).read_text())

    prefs = load_json("user_prefs.json")
    funds = load_json("fundamentals.json")
    sents = load_json("sentiments.json")
    recs = run_recommender(funds, sents, prefs)
    import pprint; pprint.pprint(recs)

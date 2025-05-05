import json
import re
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
import os


def get_recommender_chain(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    scoring_method: str = "default"
) -> LLMChain:
    """
    Returns an LLMChain configured as the RecommenderAgent.
    You can specify a scoring_method description that the LLM will follow when ranking tickers.
    """
    prompt_template = PromptTemplate(
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

Rank the top 5 tickers for this user from most to least suitable using the provided scoring method, and for each ticker give a 1-sentence rationale.
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
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    return LLMChain(llm=llm, prompt=prompt_template, output_key="recommendations_json")


def run_recommender(
    fundamentals: dict,
    sentiments: dict,
    user_preferences: dict,
    scoring_method: str = "Rank by weighted sum of sentiment * 0.5 + revenue growth * 0.3 + margin * 0.2",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2
) -> list:
    """
    Runs the recommender chain with an optional scoring_method.
    Returns parsed JSON recommendations.
    """
    chain = get_recommender_chain(
        model_name=model_name,
        temperature=temperature,
        scoring_method=scoring_method
    )
    fundamentals_json = json.dumps(fundamentals)
    sentiments_json = json.dumps(sentiments)
    user_prefs_json = json.dumps(user_preferences)

    response = chain.run(
        fundamentals_json=fundamentals_json,
        sentiments_json=sentiments_json,
        user_preferences=user_prefs_json,
        scoring_method=scoring_method
    )

    if not response or not response.strip():
        print("[run_recommender] Empty response from LLM")
        return []

    cleaned = response.strip()
    # Strip code fences if present
    m = re.search(r"```json\s*(.*?)```", cleaned, re.S)
    if m:
        cleaned = m.group(1).strip()
    else:
        m2 = re.search(r"```(.*?)```", cleaned, re.S)
        if m2:
            cleaned = m2.group(1).strip()

    try:
        recommendations = json.loads(cleaned)
    except json.JSONDecodeError:
        print("[run_recommender] Failed to parse JSON. Raw cleaned response below:")
        print(cleaned)
        raise
    return recommendations


if __name__ == "__main__":
    cwd = os.getcwd()
    def load_json_file(filename):
        path = os.path.join(cwd, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{filename} not found in working directory: {cwd}")
        with open(path) as f:
            return json.load(f)

    dummy_user_prefs = load_json_file("src/utils/user_prefs.json")
    dummy_fundamentals = load_json_file("src/utils/fundamentals.json")
    dummy_sentiments = load_json_file("src/utils/sentiments.json")

    # Optionally define a custom scoring method string
    custom_scoring = (
        "Score = sentiment_score * 0.4 + YoY_revenue_growth * 0.4 + net_profit_margin * 0.2; "
        "higher is better for each metric."
    )

    recs = run_recommender(
        fundamentals=dummy_fundamentals,
        sentiments=dummy_sentiments,
        user_preferences=dummy_user_prefs,
        scoring_method=custom_scoring
    )

    import pprint
    pprint.pprint(recs)

# router.py
# The brain of AdaptiveRAG.
# Looks at the user's question and decides
# WHERE to look for the answer.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)


class RouteDecision(BaseModel):
    """Decision on where to route the query."""
    source: str = Field(
        description="Where to look for the answer. "
                    "Choose: 'web', 'documents', or 'both'"
    )
    reason: str = Field(
        description="One sentence explaining why you chose this source"
    )


def route_query(question: str) -> RouteDecision:
    """
    Given a question, decide whether to search:
    - 'web'       → current events, news, latest info
    - 'documents' → questions about uploaded study materials
    - 'both'      → needs both current info AND document context
    """
    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at routing questions to the right source.

Given a user question, decide where to look:
- 'web': for current events, recent news, latest updates, 
         real-time information, or anything that changes frequently
- 'documents': for questions about study materials, uploaded documents,
               specific topics the user has provided notes about
- 'both': when the question needs both current web info AND 
          document context to answer properly

Return ONLY one of: web, documents, both"""),
        ("human", "Question: {question}")
    ])

    chain = prompt | structured_llm
    result = chain.invoke({"question": question})
    return result


if __name__ == "__main__":
    # Quick test
    tests = [
        "What is machine learning?",
        "What happened in AI news this week?",
        "How does the latest GPT-4 compare to what I studied?"
    ]
    for q in tests:
        decision = route_query(q)
        print(f"Q: {q}")
        print(f"→ Route: {decision.source} | Reason: {decision.reason}\n")
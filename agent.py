# agent.py
# The LangGraph agent that connects everything together.
# It routes queries, searches sources, grades results,
# and generates answers with HITL when unsure.

import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable
from router import route_query
from retriever import search_documents, has_documents

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

web_search_tool = TavilySearch(
    max_results=5,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)


class AgentState(TypedDict):
    """The state that flows through the graph."""
    question: str
    route: str
    web_results: List[dict]
    doc_results: List[dict]
    grade: str
    answer: str
    needs_clarification: bool
    clarification_question: str
    messages: List


def route_node(state: AgentState) -> AgentState:
    """Decide where to search."""
    print(f"\n🔀 Routing question: {state['question']}")
    decision = route_query(state["question"])
    print(f"   → Route: {decision.source} ({decision.reason})")
    return {**state, "route": decision.source}


def web_search_node(state: AgentState) -> AgentState:
    """Search the web using Tavily."""
    print("🌐 Searching the web...")
    raw = web_search_tool.invoke(state["question"])
    # TavilySearch returns dict with 'results' key
    if isinstance(raw, dict):
        results = raw.get("results", [])
    elif isinstance(raw, list):
        results = raw
    else:
        results = []
    print(f"   → Found {len(results)} web results")
    return {**state, "web_results": results}


def document_search_node(state: AgentState) -> AgentState:
    """Search uploaded documents."""
    print("📄 Searching documents...")
    if not has_documents():
        print("   → No documents indexed yet")
        return {**state, "doc_results": []}
    results = search_documents(state["question"])
    print(f"   → Found {len(results)} document chunks")
    return {**state, "doc_results": results}


def grade_results_node(state: AgentState) -> AgentState:
    """
    CRAG — Grade whether retrieved results actually
    answer the question before generating a response.
    """
    print("🎯 Grading results...")
    has_web = len(state.get("web_results", [])) > 0
    has_docs = len(state.get("doc_results", [])) > 0

    if not has_web and not has_docs:
        print("   → No results found!")
        return {**state, "grade": "insufficient"}

    context = ""
    if has_web:
        web_list = state["web_results"] if isinstance(state["web_results"], list) else []
        context += "\n".join([
            r.get("content", "") if isinstance(r, dict) else str(r)
            for r in web_list[:3]
        ])
    if has_docs:
        context += "\n".join([r.get("text", "") for r in state["doc_results"][:3]])

    grade_prompt = f"""Does the following context contain enough information 
to answer this question: "{state['question']}"?

Context:
{context[:2000]}

Reply with only: 'sufficient' or 'insufficient'"""

    response = llm.invoke([HumanMessage(content=grade_prompt)])
    grade = "sufficient" if "sufficient" in response.content.lower() else "insufficient"
    print(f"   → Grade: {grade}")
    return {**state, "grade": grade}


def clarify_node(state: AgentState) -> AgentState:
    """
    HITL — When results are insufficient,
    ask the user for clarification instead of guessing.
    """
    print("🤔 Asking user for clarification...")
    clarify_prompt = f"""The question "{state['question']}" couldn't be answered 
with available information. Generate ONE short clarifying question to help 
get a better answer. Be specific and helpful."""

    response = llm.invoke([HumanMessage(content=clarify_prompt)])
    return {
        **state,
        "needs_clarification": True,
        "clarification_question": response.content.strip(),
        "answer": ""
    }


def generate_answer_node(state: AgentState) -> AgentState:
    """Generate the final answer using all retrieved context."""
    print("✍️  Generating answer...")

    context_parts = []

    if state.get("web_results"):
        web_list = state["web_results"] if isinstance(state["web_results"], list) else []
        web_context = "\n".join([
            f"[Web] {r.get('content', str(r))[:500]}" if isinstance(r, dict) else f"[Web] {str(r)[:500]}"
            for r in web_list[:3]
        ])
        context_parts.append(f"WEB SEARCH RESULTS:\n{web_context}")

    if state.get("doc_results"):
        doc_context = "\n".join([
            f"[Doc: {r.get('source', 'unknown')}] {r.get('text', '')[:500]}"
            for r in state["doc_results"][:3]
        ])
        context_parts.append(f"DOCUMENT RESULTS:\n{doc_context}")

    context = "\n\n".join(context_parts)

    prompt = f"""Answer the following question using ONLY the provided context.
Always mention which source you used (web or document).
If context is insufficient, say so honestly.

Context:
{context}

Question: {state['question']}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        **state,
        "answer": response.content.strip(),
        "needs_clarification": False
    }


def should_clarify(state: AgentState) -> str:
    """Decide whether to clarify or generate answer."""
    if state.get("grade") == "insufficient":
        return "clarify"
    return "generate"


def route_to_search(state: AgentState) -> str:
    """Route to the right search node."""
    route = state.get("route", "both")
    if route == "web":
        return "web"
    elif route == "documents":
        return "documents"
    else:
        return "both"


def build_graph():
    """Build the LangGraph agent graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("router", route_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("document_search", document_search_node)
    workflow.add_node("grade", grade_results_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("generate", generate_answer_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_to_search,
        {
            "web": "web_search",
            "documents": "document_search",
            "both": "web_search",
        }
    )

    workflow.add_edge("web_search", "grade")
    workflow.add_edge("document_search", "grade")

    workflow.add_conditional_edges(
        "grade",
        should_clarify,
        {
            "clarify": "clarify",
            "generate": "generate",
        }
    )

    workflow.add_edge("clarify", END)
    workflow.add_edge("generate", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


@traceable
def run_agent(question: str, thread_id: str = "default") -> dict:
    """Run the agent on a question."""
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question": question,
        "route": "",
        "web_results": [],
        "doc_results": [],
        "grade": "",
        "answer": "",
        "needs_clarification": False,
        "clarification_question": "",
        "messages": [],
    }

    result = graph.invoke(initial_state, config)
    return result
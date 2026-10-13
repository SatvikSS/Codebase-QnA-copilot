"""The agent loop: search, judge the evidence, search again if it is thin, answer, verify.

resolve_project -> rewrite -> route -> retrieve -> rerank -> check_evidence
    -> (thin evidence: back to rewrite) -> answer -> verify -> (unsupported: back to rewrite)
"""

from dataclasses import dataclass, replace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.decisions import DecisionProvider, LLMDecisionProvider
from src.agent.tools import STRATEGIES, run_strategy
from src.kb.config import load_config
from src.rag.answer import NOT_FOUND, TOP_K, Answer, format_sources, generate_answer, source_label
from src.rag.llm import LLMResponse, complete

MAX_ATTEMPTS = 3   # searches per question
CANDIDATES = 15    # chunks fetched per search
ALL_PROJECTS = "all projects"


class AgentState(TypedDict, total=False):
    question: str
    project: str | None
    query: str
    queries: list[str]
    strategy: str
    tried: list[str]
    candidates: list[dict]
    evidence: list[dict]
    enough: bool
    answer: Answer
    supported: bool
    attempts: int
    trace: list[str]
    responses: list[LLMResponse]


@dataclass(frozen=True)
class AgentResult:
    answer: Answer
    trace: tuple[str, ...]
    responses: tuple[LLMResponse, ...]  # every LLM call: decisions, query rewrites, answer


def build_agent(provider: DecisionProvider, projects: list[str]):
    def resolve_project(state: AgentState) -> dict:
        options = sorted(projects) + [ALL_PROJECTS]
        top = provider.choose(
            state["question"],
            options,
            "Which project is this question about? Choose 'all projects' "
            "if it spans several projects or none is clearly named.",
        )[0]
        return {
            "project": None if top == ALL_PROJECTS else top,
            "trace": state["trace"] + [f"project: {top}"],
        }

    def rewrite(state: AgentState) -> dict:
        prompt = (
            "Rewrite this question as a short search query for a code search engine. "
            "Keep exact names, identifiers, file names, table and column names. "
            "Drop filler words. Return only the query.\n\n"
            f"Question: {state['question']}"
        )
        if state["queries"]:
            prompt += f"\n\nThese queries did not find enough, so use different wording: {state['queries']}"
        response = complete([{"role": "user", "content": prompt}])
        query = response.text.strip().strip('"')
        return {
            "query": query,
            "queries": state["queries"] + [query],
            "responses": state["responses"] + [response],
            "trace": state["trace"] + [f"query: {query}"],
        }

    def route(state: AgentState) -> dict:
        untried = [name for name in STRATEGIES if name not in state["tried"]] or list(STRATEGIES)
        options = [f"{name}: {STRATEGIES[name]}" for name in untried]
        top = provider.choose(
            state["question"], options, "Which search strategy fits this question best?"
        )[0]
        strategy = top.split(":")[0]
        return {"strategy": strategy, "trace": state["trace"] + [f"strategy: {strategy}"]}

    def retrieve(state: AgentState) -> dict:
        # If the first search did not work, the project guess may be wrong, so search everywhere
        project = state["project"] if state["attempts"] == 0 else None
        found = run_strategy(state["strategy"], state["query"], project, CANDIDATES)
        merged = list({chunk["id"]: chunk for chunk in state["candidates"] + found}.values())
        return {
            "candidates": merged,
            "attempts": state["attempts"] + 1,
            "tried": state["tried"] + [state["strategy"]],
            "trace": state["trace"] + [f"retrieved {len(found)} chunks ({len(merged)} candidates)"],
        }

    def rerank(state: AgentState) -> dict:
        chunks = state["candidates"]
        scores = provider.score(
            state["question"],
            [f"{source_label(chunk)}\n{chunk['text']}" for chunk in chunks],
            "Rate how useful each item is for answering the question. "
            "3 = contains the answer, 2 = closely related, 1 = loosely related, 0 = irrelevant.",
        )
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:TOP_K]  # stable: ties keep search order
        evidence = [chunks[i] for i in order]
        top = ", ".join(f"{source_label(chunks[i])} ({scores[i]})" for i in order[:3])
        return {"evidence": evidence, "trace": state["trace"] + [f"rerank top 3: {top}"]}

    def check_evidence(state: AgentState) -> dict:
        enough = provider.judge(
            state["question"],
            f"Sources:\n\n{format_sources(state['evidence'])}",
            "Do these sources contain enough information to answer the question?",
        )
        return {"enough": enough, "trace": state["trace"] + [f"enough evidence: {enough}"]}

    def answer(state: AgentState) -> dict:
        result = generate_answer(state["question"], state["evidence"])
        return {"answer": result, "responses": state["responses"] + [result.response]}

    def verify(state: AgentState) -> dict:
        result = state["answer"]
        if result.text.strip() == NOT_FOUND:
            return {"supported": True, "trace": state["trace"] + ["answer: not found"]}

        # All sources keep the numbers the answer cites them by, so [3] means the same thing here
        supported = bool(result.citations) and provider.judge(
            state["question"],
            f"Sources:\n\n{format_sources(state['evidence'])}\n\nAnswer:\n{result.text}",
            "Is each claim in the answer supported by the source(s) it cites with [n]? "
            "Answer true if the main claims are supported; minor wording differences are fine.",
        )
        update = {"supported": supported, "trace": state["trace"] + [f"citations support answer: {supported}"]}
        if not supported and state["attempts"] >= MAX_ATTEMPTS:
            update["answer"] = replace(result, text=NOT_FOUND, citations=())
        return update

    def after_check(state: AgentState) -> str:
        return "answer" if state["enough"] or state["attempts"] >= MAX_ATTEMPTS else "rewrite"

    def after_verify(state: AgentState) -> str:
        return END if state["supported"] or state["attempts"] >= MAX_ATTEMPTS else "rewrite"

    graph = StateGraph(AgentState)
    for node in (resolve_project, rewrite, route, retrieve, rerank, check_evidence, answer, verify):
        graph.add_node(node.__name__, node)

    graph.add_edge(START, "resolve_project")
    graph.add_edge("resolve_project", "rewrite")
    graph.add_edge("rewrite", "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "check_evidence")
    graph.add_conditional_edges("check_evidence", after_check, {"answer": "answer", "rewrite": "rewrite"})
    graph.add_edge("answer", "verify")
    graph.add_conditional_edges("verify", after_verify, {END: END, "rewrite": "rewrite"})
    return graph.compile()


def run_agent(question: str, provider: DecisionProvider | None = None) -> AgentResult:
    provider = provider or LLMDecisionProvider()
    app = build_agent(provider, sorted(load_config().include_projects))
    state = app.invoke(
        {
            "question": question,
            "queries": [],
            "tried": [],
            "candidates": [],
            "attempts": 0,
            "trace": [],
            "responses": [],
        },
        {"recursion_limit": 50},
    )
    decision_calls = getattr(provider, "responses", [])
    return AgentResult(
        answer=state["answer"],
        trace=tuple(state["trace"]),
        responses=tuple(state["responses"] + decision_calls),
    )

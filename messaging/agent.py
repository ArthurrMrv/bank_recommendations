"""LangGraph agent: write a message as Claire, score it (checks + LLM judge), retry up to twice, else flag."""
import json
import os
import re
from pathlib import Path
from string import Template
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from messaging import checks

HERE = Path(__file__).parent
MIN_JUDGE = 4
MAX_ATTEMPTS = 3  # first draft + 2 retries
JUDGE_CRITERIA = ("tone_match", "motivation_framing", "channel_fit", "language_correct", "clarity", "no_overpromise",
                  "factual_accuracy")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


class State(TypedDict, total=False):
    facts: dict
    prompt: str
    draft: str | None
    scores: list[dict]  # one grid per attempt: criterion -> {score, passed, note}
    attempts: int
    approved: bool
    flagged: bool


def render(name, **values):
    return Template((HERE / name).read_text()).safe_substitute(**values)


def feedback(state):
    if not state.get("scores"):
        return ""
    failed = "\n".join(f"- {k} (score {v['score']}): {v['note']}" for k, v in state["scores"][-1].items()
                       if not v["passed"])
    return (f"\n# Your previous draft was rejected\n\n```\n{state['draft']}\n```\n\nFailed criteria:\n{failed}\n\n"
            "Rewrite the message so every criterion passes, still following all the rules above.\n")


def parse_judge(text):
    try:
        raw = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        return {k: {"score": int(raw[k]["score"]), "passed": int(raw[k]["score"]) >= MIN_JUDGE,
                    "note": str(raw[k].get("reason", ""))} for k in JUDGE_CRITERIA}
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        return {"judge": {"score": 0, "passed": False, "note": f"unparseable judge output: {e}"}}


def build_graph(writer, judge):
    def generate(state):
        f = state["facts"]
        prompt = render("prompt.md", **f, max_chars=checks.MAX_CHARS[f["channel"]], feedback=feedback(state))
        return {"prompt": prompt, "draft": writer.invoke(prompt).content.strip(),
                "attempts": state.get("attempts", 0) + 1}

    def score(state):
        f, msg = state["facts"], state["draft"]
        grid = {k: {"score": int(ok), "passed": ok, "note": note} for k, (ok, note) in checks.run(msg, f).items()}
        grid |= parse_judge(judge.invoke(render("judge.md", **f, message=msg)).content)
        return {"scores": state.get("scores", []) + [grid]}

    def route(state):
        if all(v["passed"] for v in state["scores"][-1].values()):
            return "approve"
        return "generate" if state["attempts"] < MAX_ATTEMPTS else "flag"

    g = StateGraph(State)
    g.add_node("generate", generate)
    g.add_node("score", score)
    g.add_node("approve", lambda s: {"approved": True, "flagged": False})
    g.add_node("flag", lambda s: {"approved": False, "flagged": True, "draft": None})
    g.add_edge(START, "generate")
    g.add_edge("generate", "score")
    g.add_conditional_edges("score", route, ["approve", "generate", "flag"])
    g.add_edge("approve", END)
    g.add_edge("flag", END)
    return g.compile()


def openrouter_models():
    from langchain_openai import ChatOpenAI

    key, model = os.environ["OPENROUTER_API_KEY"], os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    writer = ChatOpenAI(model=model, base_url=OPENROUTER_URL, api_key=key, temperature=0.7)
    judge = ChatOpenAI(model=os.environ.get("OPENROUTER_JUDGE_MODEL", model), base_url=OPENROUTER_URL,
                       api_key=key, temperature=0)
    return writer, judge


def write_message(facts, writer, judge):
    return build_graph(writer, judge).invoke({"facts": facts, "attempts": 0, "scores": []})

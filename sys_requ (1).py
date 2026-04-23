import os
import json
import logging
from typing import TypedDict, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential


# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# state definition
class AgentState(TypedDict):
    current_section: str
    sections: List[str]
    document: Dict[str, str]
    messages: List[str]
    user_input: str
    completed: bool


# llm setup
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=api_key
    )


llm = get_llm()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def safe_llm_call(prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        raise


# prompts
def question_prompt(section, context):
    return f"""
You are a senior system analyst.

Section: {section}

Context:
{json.dumps(context, indent=2)}

Generate 3 precise, implementation-level questions.
Output as numbered list only.
"""


def refine_prompt(section, user_input):
    return f"""
Convert into structured requirements.

Section: {section}

Rules:
- Remove ambiguity
- Use bullet points
- Make it developer-ready

Input:
{user_input}
"""


def validation_prompt(document):
    return f"""
Perform validation:
- Remove redundancy
- Detect missing parts
- Highlight risks

Document:
{json.dumps(document, indent=2)}
"""


def feature_prompt(document):
    return f"""
Break into features:
- Independent modules
- Include dependencies

Document:
{json.dumps(document, indent=2)}
"""

# nodes
def ask_question(state: AgentState):
    prompt = question_prompt(state["current_section"], state["document"])
    response = safe_llm_call(prompt)

    logger.info(f"\n[{state['current_section']}] Questions:\n{response}")

    state["messages"].append(response)
    return state


def collect_input(state: AgentState):
    user_input = input(f"\nEnter details for {state['current_section']}: ")
    state["user_input"] = user_input
    return state


def refine_section(state: AgentState):
    prompt = refine_prompt(state["current_section"], state["user_input"])
    response = safe_llm_call(prompt)

    state["document"][state["current_section"]] = response
    return state


def next_section(state: AgentState):
    idx = state["sections"].index(state["current_section"])

    if idx + 1 < len(state["sections"]):
        state["current_section"] = state["sections"][idx + 1]
    else:
        state["completed"] = True

    return state


def validate_document(state: AgentState):
    validation = safe_llm_call(validation_prompt(state["document"]))
    features = safe_llm_call(feature_prompt(state["document"]))

    state["messages"].append("\nVALIDATION:\n" + validation)
    state["messages"].append("\nFEATURES:\n" + features)

    return state


# graph construction
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("ask", ask_question)
    graph.add_node("input", collect_input)
    graph.add_node("refine", refine_section)
    graph.add_node("next", next_section)
    graph.add_node("validate", validate_document)

    graph.set_entry_point("ask")

    graph.add_edge("ask", "input")
    graph.add_edge("input", "refine")
    graph.add_edge("refine", "next")

    graph.add_conditional_edges(
        "next",
        lambda state: "validate" if state["completed"] else "ask"
    )

    graph.add_edge("validate", END)

    return graph.compile()


# memory store
class MemoryStore:
    def __init__(self, filepath="state.json"):
        self.filepath = filepath

    def save(self, state):
        try:
            with open(self.filepath, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Save failed: {e}")

    def load(self) -> Optional[dict]:
        if not os.path.exists(self.filepath):
            return None
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return None


# output generation
def generate_markdown(document, messages):
    md = "# Requirements Document\n\n"

    for section, content in document.items():
        md += f"## {section}\n{content}\n\n"

    md += "## Analysis\n\n"
    md += "\n\n".join(messages)

    return md


# main execution
def main():
    sections = [
        "Project Overview",
        "Actors",
        "Functional Requirements",
        "Non-Functional Requirements",
        "Constraints",
        "Assumptions",
        "Dependencies"
    ]

    state = {
        "current_section": sections[0],
        "sections": sections,
        "document": {},
        "messages": [],
        "user_input": "",
        "completed": False
    }

    store = MemoryStore()
    graph = build_graph()

    final_state = graph.invoke(state)

    store.save(final_state)

    md = generate_markdown(
        final_state["document"],
        final_state["messages"]
    )

    with open("requirements.md", "w") as f:
        f.write(md)

    print("\n✅ requirements.md generated successfully")


if __name__ == "__main__":
    main()

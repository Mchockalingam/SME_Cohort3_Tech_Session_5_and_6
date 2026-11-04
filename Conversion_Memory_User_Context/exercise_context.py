"""Seven-minute exercise: complete ONLY the two TODO functions.

Run: python exercise_context.py
All fixtures and graph wiring are supplied. No API key or LLM is used.
"""

from dataclasses import dataclass
import json

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass(frozen=True)
class UserContext:
    tenant_id: str
    user_id: str


def memory_namespace(context: UserContext) -> tuple[str, ...]:
    # TODO 1: Return a five-part tuple in this order:
    # "tenants", the tenant ID, "users", the user ID, "preferences".
    raise NotImplementedError("Complete TODO 1: scope memory by tenant AND user.")


def read_response_style(store, namespace: tuple[str, ...]) -> str:
    # TODO 2: Get key "response_style" from this namespace.
    # Missing record -> "default". Otherwise read item.value["value"].
    # Accept only "brief" or "detailed"; any other value -> "default".
    raise NotImplementedError("Complete TODO 2: retrieve a preference with a fallback.")


def build_graph(store):
    def assistant(state: MessagesState, runtime: Runtime[UserContext]):
        namespace = memory_namespace(runtime.context)
        style = read_response_style(runtime.store, namespace)
        return {"messages": [AIMessage(content=f"response_style={style}")]}

    builder = StateGraph(MessagesState, context_schema=UserContext)
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    return builder.compile(checkpointer=InMemorySaver(), store=store)


def main():
    store = InMemoryStore()
    # Fixtures are explicit so an incorrect namespace does not also change
    # the seed data and make a broken solution appear correct.
    store.put(("tenants", "acme", "users", "ananya", "preferences"),
              "response_style", {"value": "brief"})
    store.put(("tenants", "acme", "users", "invalid", "preferences"),
              "response_style", {"value": "not_an_allowed_style"})
    graph = build_graph(store)
    scenarios = [
        ("Ananya, first conversation", UserContext("acme", "ananya"), "one", "brief"),
        ("Ananya, NEW conversation", UserContext("acme", "ananya"), "two", "brief"),
        ("Ravi, same company", UserContext("acme", "ravi"), "one", "default"),
        ("Ananya ID, different company", UserContext("beta", "ananya"), "one", "default"),
        ("Invalid saved value", UserContext("acme", "invalid"), "one", "default"),
    ]
    for label, context, conversation_id, expected in scenarios:
        # Production servers must additionally verify authentication/ownership.
        thread_id = json.dumps([context.tenant_id, context.user_id, conversation_id])
        result = graph.invoke(
            {"messages": [HumanMessage(content="Help with my printer.")]},
            config={"configurable": {"thread_id": thread_id}},
            context=context,
        )
        actual = result["messages"][-1].content
        assert actual == f"response_style={expected}", f"{label}: got {actual!r}"
        assert len(result["messages"]) == 2, "Every scenario should have its own transcript."
        print(f"PASS: {label} -> {actual}")
    print("PASS: all five exercise scenarios. Explain WHY the namespaces differ.")


if __name__ == "__main__":
    main()

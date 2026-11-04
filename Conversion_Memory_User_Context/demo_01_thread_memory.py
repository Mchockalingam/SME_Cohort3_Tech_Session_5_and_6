"""Demo 1: conversation memory in a real LangGraph, without an API key.

Run: python demo_01_thread_memory.py
The reply is a deterministic teaching simulation, NOT a language model.
LangGraph manages the state, message updates, threads, and checkpoints.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


def find_operating_system(messages):
    """Read only user messages; the latest explicit OS mention wins."""
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        text = str(message.content).casefold()
        if "windows 11" in text:
            return "Windows 11"
        if "ubuntu" in text:
            return "Ubuntu"
    return None


def assistant(state: MessagesState):
    """A node reads the assembled thread state and returns one new message."""
    latest_text = str(state["messages"][-1].content).casefold()
    operating_system = find_operating_system(state["messages"])
    if "which operating system" in latest_text:
        if operating_system:
            reply = f"You mentioned {operating_system}."
        else:
            reply = "I cannot find an operating system in this conversation."
    else:
        reply = "Thanks. Ask me which operating system you mentioned."
    # MessagesState merges this update. Do not return the entire old history.
    return {"messages": [AIMessage(content=reply)]}


def build_graph(checkpointer=None):
    """Build the same workflow with or without a checkpoint backend."""
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    return builder.compile(checkpointer=checkpointer)


def show_turn(graph, text, config):
    # Send only the new user message. A checkpointer restores prior state.
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
    print(f"User: {text}")
    print(f"Simulated assistant: {result['messages'][-1].content}")
    print(f"Messages in returned state: {len(result['messages'])}")
    return result


def main():
    print("DEMO 1 | Real LangGraph memory; simulated replies; no LLM calls.\n")
    same_thread = {"configurable": {"thread_id": "it-support-session-a"}}
    new_thread = {"configurable": {"thread_id": "it-support-session-b"}}
    statement = "I use Windows 11."
    question = "Which operating system did I mention?"

    print("A. No checkpointer: a thread ID alone does not retain history.")
    stateless = build_graph()
    show_turn(stateless, statement, same_thread)
    forgotten = show_turn(stateless, question, same_thread)
    assert "Windows 11" not in forgotten["messages"][-1].content
    assert len(forgotten["messages"]) == 2

    print("\nB. InMemorySaver: the same thread resumes its conversation.")
    graph = build_graph(checkpointer=InMemorySaver())
    show_turn(graph, statement, same_thread)
    remembered = show_turn(graph, question, same_thread)
    assert "Windows 11" in remembered["messages"][-1].content
    assert len(remembered["messages"]) == 4

    print("\nC. A different thread starts with a separate conversation.")
    separate = show_turn(graph, question, new_thread)
    assert "Windows 11" not in separate["messages"][-1].content
    assert len(separate["messages"]) == 2

    snapshot = graph.get_state(same_thread)
    roles = ", ".join(message.type for message in snapshot.values["messages"])
    print(f"\nOriginal thread still contains 4 messages: [{roles}]")
    assert len(snapshot.values["messages"]) == 4
    print("PASS: no memory without a saver; same-thread recall; thread separation.")
    print("InMemorySaver lives in RAM. Its data is lost when this process ends.")
    print("Thread IDs select records; a real app must also authorize access.")


if __name__ == "__main__":
    main()

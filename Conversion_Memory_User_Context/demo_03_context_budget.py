"""Demo 3: a smaller model input is different from deleting stored messages.

Uses a real LangGraph workflow and LangChain's message trimming utility.
Replies and the small token budget are simulated for classroom visibility.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


# These are deliberately small illustrative allowances, NOT model specifications.
CONTEXT_ALLOWANCE = 384
OUTPUT_ALLOWANCE = 96
SAFETY_MARGIN = 32
SYSTEM = SystemMessage(content="You are an IT support assistant. Use only supplied facts.")
HISTORY_ALLOWANCE = (
    CONTEXT_ALLOWANCE - OUTPUT_ALLOWANCE - SAFETY_MARGIN
    - count_tokens_approximately([SYSTEM])
)


def select_history(messages, allowance=HISTORY_ALLOWANCE):
    recent = trim_messages(
        messages,
        max_tokens=allowance,
        strategy="last",
        token_counter=count_tokens_approximately,
        start_on="human",
        end_on="human",  # This demo has only human/AI messages and no tools.
        allow_partial=False,
    )
    if not recent:
        raise ValueError("The newest message is too long; ask for a shorter request.")
    return recent


def assistant(state: MessagesState):
    full_history = state["messages"]
    selected = select_history(full_history)
    model_input = [SYSTEM, *selected]
    print(f"Stored messages BEFORE this reply: {len(full_history)}")
    print(f"Selected history messages: {len(selected)}")
    print(f"History allowance, approximate tokens: {HISTORY_ALLOWANCE}")
    print(f"Full history, approximate tokens: {count_tokens_approximately(full_history)}")
    print(f"Selected history, approximate tokens: {count_tokens_approximately(selected)}")
    print(f"Model input including system, approximate tokens: {count_tokens_approximately(model_input)}")
    print("Selected input roles:", [m.type for m in model_input])

    # Demonstrates the INFORMATION available to a model; this is not an LLM.
    os_available = any("Windows 11" in str(m.content) for m in selected)
    reply = (
        "[Simulated] You mentioned Windows 11."
        if os_available
        else "[Simulated] Your operating system is absent from my selected context. Please repeat it."
    )
    print("Assistant:", reply)
    assert len(selected) < len(full_history)
    assert selected[-1].content == full_history[-1].content
    assert count_tokens_approximately(selected) <= HISTORY_ALLOWANCE
    assert not os_available, "The classroom example should visibly lose the early OS fact."
    # Return ONLY the reply. Do not replace stored messages with `selected`.
    return {"messages": [AIMessage(content=reply)]}


def main():
    print("MODE: offline, simulated reply; approximate classroom token budget\n")
    # Seed a fictional conversation ONCE so a long history is visible immediately.
    history = [HumanMessage(content="I use Windows 11."), AIMessage(content="Understood.")]
    for number in range(1, 7):
        history.extend([
            HumanMessage(content=f"Status {number}: " + "The print queue is still being checked. " * 6),
            AIMessage(content="We will keep checking the printer status. " * 6),
        ])
    history.append(HumanMessage(content="Which operating system did I mention?"))
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "context-budget-demo"}}
    graph.invoke({"messages": history}, config=config)
    stored = graph.get_state(config).values["messages"]
    print(f"Stored messages AFTER this reply: {len(stored)}")
    assert len(stored) == len(history) + 1
    assert any("Windows 11" in str(m.content) for m in stored)
    print("PASS: early fact remains in stored state even though it was excluded from model input.")
    print("Discussion: preserve important task facts or a checked summary when recency alone loses them.")


if __name__ == "__main__":
    main()

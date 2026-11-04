"""Optional extension: retain a conversation across separate Python processes.

Run these as separate terminal commands, in order:
  python optional_sqlite_restart.py write
  python optional_sqlite_restart.py read
  python optional_sqlite_restart.py reset

Only synthetic workshop data is used. This is a local-development example.
"""

import argparse
import os
from pathlib import Path

# Set the framework's restricted checkpoint-deserialization mode before imports.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from demo_01_thread_memory import build_graph


THREAD_ID = "workshop-sqlite-os-example"
CONFIG = {"configurable": {"thread_id": THREAD_ID}}
DEFAULT_DB = Path(__file__).resolve().parent / ".workshop_data" / "checkpoints.sqlite"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "read", "reset"))
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    db_path = args.db.expanduser().resolve()

    # Do not create an empty database when a learner has not run "write" yet.
    if args.action == "read" and not db_path.exists():
        parser.error("No checkpoint database yet. Run the 'write' command first.")
    if args.action == "reset" and not db_path.exists():
        print("Nothing to reset: the workshop checkpoint database does not exist.")
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # This context manager opens and closes the database connection correctly.
    # SQLite setup is automatic. Run the graph while the saver remains open.
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        if args.action == "reset":
            saver.delete_thread(THREAD_ID)
            print("Deleted checkpoints for this workshop thread only.")
            print("The database file and any other threads are unchanged.")
            return

        graph = build_graph(checkpointer=saver)
        if args.action == "write":
            text = "I use Windows 11."
        else:
            if not graph.get_state(CONFIG).values.get("messages"):
                parser.error("No saved workshop thread. Run the 'write' command first.")
            text = "Which operating system did I mention?"

        result = graph.invoke({"messages": [HumanMessage(content=text)]}, CONFIG)
        reply = result["messages"][-1].content
        print(f"User: {text}")
        print(f"Simulated assistant: {reply}")
        print(f"Checkpoint database: {db_path}")
        if args.action == "write":
            print("Process finished writing. Now run 'read' as a separate command.")
        else:
            assert "Windows 11" in reply
            print("PASS: a new Python process recovered the earlier conversation.")
        print("Persistence comes from the checkpoint backend, not the model.")
        print("This example does not create a durable cross-thread user profile.")


if __name__ == "__main__":
    main()

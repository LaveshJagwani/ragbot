from collections import defaultdict

MAX_TURNS = 5
memory_store = defaultdict(list)  # session_id → [(q, a)]

def get_history(session_id: str):
    return memory_store[session_id]

def append_history(session_id: str, question: str, answer: str):
    history = memory_store[session_id]
    history.append((question, answer))

    if len(history) > MAX_TURNS:
        history.pop(0)

def reset_session(session_id: str):
    memory_store.pop(session_id, None)

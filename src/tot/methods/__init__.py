def get_method(name):
    if name == "bfs":
        from .bfs import solve
    elif name == "dfs":
        from .dfs import solve
    elif name == "cot":
        from .cot import solve
    elif name == "bot":
        from .bot import solve
    elif name == "simple_dfs":
        from .simple_dfs import solve
    elif name == "dtree":
        from .dtree import solve
    elif name == "pure_llm":
        from .pure_llm import solve
    else:
        raise ValueError(f"Unknown method: {name}")

    return type('Method', (), {'solve': staticmethod(solve)})()

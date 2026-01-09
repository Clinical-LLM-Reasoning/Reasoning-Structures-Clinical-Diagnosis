def get_task(name, **kwargs):
    if name == "thyroid_lab":
        from .thyroid_lab_task import ThyroidLabTask
        return ThyroidLabTask(**kwargs)
    else:
        raise ValueError(f"Unknown task: {name}")

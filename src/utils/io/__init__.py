import os


def get_filename(p: str):
    return os.path.splitext(os.path.split(p)[-1])[0]

import signal


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Timeout reached!")


class TimeoutManager:
    def __init__(self, seconds: int = 30):
        self._seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self._seconds)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
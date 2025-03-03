from time import time
from typing import ContextManager


class Timed(ContextManager):
    """Timer context manager

    Usage:
    with Timed():
        # code to be timed

    or

    output_dict = {}
    with Timed("job_name", output_dict):
        # code to be timed
    """

    def __init__(self, label: str = None, output_to: dict = None):
        """
        Args:
            label (str): Label to be printed or stored in the output_dict
            output_to (dict): Dictionary to store the time taken, if None,
                the time is printed
        """
        self.label = label or "Time taken"
        self.output_dict = output_to

        if label is not None and output_to is not None:
            self.print = False
        else:
            self.print = True

        self.start = None
        self.end = None

    def go(self):
        self.start = time()
        return self

    def stop(self):
        self.end = time()
        if self.print:
            seconds = self.end - self.start
            if seconds < 60:
                print(f"{self.label}: {seconds:.2f}s ")
            else:
                print(f"{self.label}: {int(seconds / 60)}m {seconds % 60:.2f}s ")
        else:
            self.output_dict[self.label] = self.end - self.start

    def __enter__(self):
        return self.go()

    def __exit__(self, *args):
        self.stop()
        return False

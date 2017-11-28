import sys


class EmopyLogger(object):
    def __init__(self, output_files=[sys.stdout]):
        for f in output_files:
            f.write("hello world")

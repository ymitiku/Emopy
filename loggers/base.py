from __future__ import print_function
import sys


class EmopyLogger(object):
    def __init__(self, output_files=[sys.stdout]):
        self.output_files = output_files
    def log(self,string):
        for f in self.output_files:
            if f == sys.stdout:
                print(string)
            elif type(f) == str:
                with open(f,"a+") as out_file:
                    out_file.write(string+"\n")
    def add_log_file(self,log_file):
        self.output_files.append(log_file)
            

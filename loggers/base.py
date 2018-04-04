from __future__ import print_function
import sys
import os
from train_config import DATA_SET_DIR,LOG_DIR, BATCH_SIZE,EPOCHS,LEARNING_RATE
import numpy as np
import time


class EmopyLogger(object):
    def __init__(self, output_files=[sys.stdout]):
        assert type(output_files)==list,"output files should be list"
        self.output_files = output_files
    def log(self,string):
        
        for f in self.output_files:
            if f == sys.stdout:
                print(string)
            elif type(f) == str:
                parent, model_file  = os.path.split(f)
                if not os.path.exists(parent):
                    
                    print("Log file directory does not exists creating now...")
                    os.mkdir(parent)
                with open(f,"a+") as out_file:
                    out_file.write(string+"\n")
    def add_log_file(self,log_file):
        self.output_files.append(log_file)
    def log_model(self,args,score,anthor):
        parent, model_file  = os.path.split(args.model_path)
        if not os.path.exists(parent):
            if args.verbose:
                print("Log file directory does not exists creating now...")
            os.mkdir(parent)
        model_number = np.fromfile(os.path.join(parent,"model_number.txt"),dtype=int)
        model_file_name = model_file+"-"+str(model_number[0]-1)
    
        self.log("**************************************")
        self.log("Trained model "+model_file_name+".json")
        self.log(time.strftime("%A %B %d,%Y %I:%M%p"))
        self.log("Dataset dir: "+DATA_SET_DIR)
        self.log("Parameters")
        self.log("_______________________________________")
        self.log("Batch-Size    : "+str(BATCH_SIZE))
        self.log("Epoches       : "+str(EPOCHS))
        self.log("Learning rate : "+str(LEARNING_RATE))
        self.log("_______________________________________")
        self.log("Loss          : "+str(score[0]))
        self.log("Accuracy      : "+str(score[1]))
        self.log("**************************************")
            

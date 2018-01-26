from runners import start_train_program,start_test_program
from train_config import DATA_SET_DIR,EPOCHS,LEARNING_RATE,STEPS_PER_EPOCH,BATCH_SIZE,AUGMENTATION
from config import SESSION
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",default=DATA_SET_DIR,type=str)
    parser.add_argument("--train",default=(SESSION=="train"),type=bool)
    parser.add_argument("--epochs",default=EPOCHS,type=int)
    parser.add_argument("--batch_size",default=BATCH_SIZE,type=int)
    parser.add_argument("--lr",default=LEARNING_RATE,type=float)
    parser.add_argument("--steps",default = STEPS_PER_EPOCH,type=int)
    parser.add_argument("--augmentation",default = AUGMENTATION,type=bool)


    args = parser.parse_args()
    if not os.path.exists(args.dataset_dir):
        print "Dataset path given does not exists"
        exit(0)

    if args.train:
        start_train_program(dataset_dir=args.dataset_dir,epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,steps=args.steps,augmentation=AUGMENTATION)
    else:
        start_test_program()

if __name__== "__main__":
    main()
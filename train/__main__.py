from train import get_cmd_args,get_network




def main():
    args = get_cmd_args()
    net = get_network(args)
    net.train(args)

if __name__ == '__main__':
    main()
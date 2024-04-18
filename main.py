import argparse
# Create the parser
import logging


from utils import *
import gc
import torch
from train import *
from adv_train import *

import math


OPTIMIZERS = ['adam', 'sgd']
SPLITS = ['iid','non_iid']
Datasets =['lisa',"eurosat"]
def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='adam')
    parser.add_argument('--dataset_root',
                        help='Dataset Root',
                        type=str,
                        default='')
    parser.add_argument('--data_split',
                        help='Data Split type;',
                        type=str,
                        choices=SPLITS,
                        default='non_iid')

    parser.add_argument('--num_rounds',
                        help='number of FL communication rounds;',
                        type=int,
                        default=400)

    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=4)
    parser.add_argument('--num_epochs', 
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=50)

    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.003)
    parser.add_argument('--ways',
                        help='meta learning ways;',
                        type=float,
                        default=5)
    parser.add_argument('--shots',
                        help='number of examples per class in each task for Meta learning; e.g., "5-shot" learning uses 5 examples per class',
                        type=float,
                        default=5)
    parser.add_argument('--meta_lr',
                        help='Meta learning rate;',
                        type=float,
                        default=0.01)

    parser.add_argument('--poison',
                        help='Poison level in Percentage',
                        type=int,
                        default=0)
    
    parser.add_argument('--lambda',
                        help='For Consistency Regularization',
                        type=int,
                        default=0)

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))



    # load selected model

    return parsed


def main():

    parsed =  read_options()
    
    poison_level = parsed["poison"]
    num_clients= parsed["clients_per_round"]
    poison_clients =  math.ceil( num_clients * (poison_level/100))
    print("Poisoned CLients:",poison_clients)
    if(poison_clients> num_clients ):
        logging.error("Poison level can not be greater than 100 %")
    elif(poison_clients>0):
        train_attack(batch_size= parsed["batch_size"]
     ,poison=poison_clients,data_split=parsed["data_split"],optimizer=parsed["optimizer"],comm_rounds = parsed["num_rounds"], local_epochs= parsed["num_epochs"], 
     lr= parsed["learning_rate"], num_clients= parsed["clients_per_round"],     ways = parsed["ways"],     meta_lr = parsed["meta_lr"], shots = parsed["shots"] , dataset_root = parsed["dataset_root"] )
    
    else: 
        adv_train(batch_size= parsed["batch_size"]
     ,poison=poison_clients,data_split=parsed["data_split"],optimizer=parsed["optimizer"],comm_rounds = parsed["num_rounds"], local_epochs= parsed["num_epochs"], 
     lr= parsed["learning_rate"], num_clients= parsed["clients_per_round"],     ways = parsed["ways"],     meta_lr = parsed["meta_lr"], shots = parsed["shots"] , dataset_root = parsed["dataset_root"], lambda_val= parsed["lambda"])
        




if __name__ == '__main__':
  
    gc.collect()
    torch.cuda.empty_cache()
    main()
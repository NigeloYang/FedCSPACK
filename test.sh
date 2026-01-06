############################################################## Test Program ##############################################################

############################################################## MNIST ##############################################################
############################################################## IID ##############################################################
python main.py --algorithm=FedAvg    --dataset=mnist --niid=iid --partition=pat --balance=balance --num_classes=10 --global_epoch=2 --ran_rate=0.3
python main.py --algorithm=FedCSPack --dataset=mnist --niid=iid --partition=pat --balance=balance --num_classes=10 --global_epoch=2 --ran_rate=0.3


############################################################## Non-IID DIR=0.05 ##############################################################
python main.py --algorithm=FedAvg    --dataset=mnist --niid=noniid --partition=dir --balance=balance --num_classes=10 --global_epoch=2 --ran_rate=0.3 --dir_alpha=0.05 
python main.py --algorithm=FedCSPack --dataset=mnist --niid=noniid --partition=dir --balance=balance --num_classes=10 --global_epoch=2 --ran_rate=0.3 --dir_alpha=0.05 
python generate_MNIST.py --niid=iid    --partition=pat   --balance=balance   --num_clients=20 --class_per_client=3 --batch_size=16 --train_ratio=0.75
python generate_MNIST.py --niid=iid    --partition=pat   --balance=unbalance --num_clients=20 --class_per_client=3 --batch_size=16 --train_ratio=0.75

# python generate_MNIST.py --niid=noniid --partition=pat   --balance=balance   --num_clients=20 --class_per_client=3 --batch_size=16 --train_ratio=0.75
# python generate_MNIST.py --niid=noniid --partition=pat   --balance=unbalance --num_clients=20 --class_per_client=3 --batch_size=16 --train_ratio=0.75

python generate_MNIST.py --niid=noniid --partition=dir   --balance=balance   --num_clients=20 --class_per_client=3 --batch_size=16 --train_ratio=0.75 --alpha=0.05
# python generate_MNIST.py --niid=noniid --partition=dir   --balance=balance   --num_clients=20 --class_per_client=3 --batch_size=16 --train_ratio=0.75 --alpha=0.1
# python generate_MNIST.py --niid=noniid --partition=dir   --balance=balance   --num_clients=20 --class_per_client=3 --batch_size=16 --train_ratio=0.75 --alpha=0.5
# python generate_MNIST.py --niid=noniid --partition=dir   --balance=balance   --num_clients=20 --class_per_client=3 --batch_size=16 --train_ratio=0.75 --alpha=1.0

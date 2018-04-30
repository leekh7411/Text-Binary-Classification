import argparse

def get_config():
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=50)
    args.add_argument('--w2v_size',type=int, default=100)
    args.add_argument('--embedding', type=int, default=16)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--lr',type=float,default=0.01)
    args.add_argument('--decay',type=float,default=0.9)
    args.add_argument('--test_set_rate',type=float,default=0.8)
    cfg = args.parse_args()
    return cfg
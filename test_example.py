import debugpy
import torch
import torchvision
import argparse
def test_msg(msg):
    assert msg == "hello world"
    print(msg)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, help="config file")
    parser.add_argument('--lr', required=False, type=str, help="learning rate")

    args, non_added = parser.parse_known_args()
    print(args)
    
    msg = "hello world"
    test_msg(msg)
    
    FCNN = torch.nn.Linear(10,3)

    target = torch.ones(1,3,dtype=torch.float32)
    loss_func = torch.nn.MSELoss()
    a = torch.randn(1,10)
    optim = torch.optim.Adam(FCNN.parameters(),lr=float(args.lr))
    
    for i in range(100):
        optim.zero_grad()
        b = FCNN(a)
        l = loss_func(b,target)
        l.backward()
        optim.step()
        
    print("end of debug")
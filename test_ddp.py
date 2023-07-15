import os
import torch.multiprocessing as mp
import numpy
import PIL
import PIL.Image
import time
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from run import Network
arguments_strOne = './images/one.png'
arguments_strTwo = './images/two.png'

torch.set_grad_enabled(True)

def use_ddp(rank, world_size):
    loss_fn = nn.MSELoss()
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model = Network().to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model_optimizer = optim.Adam(ddp_model.parameters(), 0.001)
    tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    
    for epoch in tqdm(range(100)):
        tenPreprocessedOne = tenOne.to(rank).view(1, 3, intHeight, intWidth)
        tenPreprocessedTwo = tenTwo.to(rank).view(1, 3, intHeight, intWidth)
        out = ddp_model(tenPreprocessedOne, tenPreprocessedTwo)
        loss = loss_fn(out, torch.randn_like(out).to(rank))
        loss.backward()
        model_optimizer.step()
        
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 4
    t1 = time.time()
    mp.spawn(use_ddp,
        args=(world_size,),
        nprocs=world_size,
        join=True)
    
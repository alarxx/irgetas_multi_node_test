import datetime, os, socket
import torch, torch.distributed as dist


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    print(f"rank: {rank}, local_rank: {local_rank}, world_size: {world_size}, device: {device}", flush=True)

    dist.init_process_group(
        "nccl", 
        device_id=device,
        timeout=datetime.timedelta(minutes=2)
    )

    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()

        x = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(x)

        expected = world_size * (1 + world_size) / 2 # sum of arithmetic progression (1+2+3+4=4*5/2)
        assert x.item() == expected, f"got {x.item()}, expected {expected}"

        print(f"{socket.gethostname()} rank={rank} local_rank={local_rank} result={x.item()} OK", flush=True)

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
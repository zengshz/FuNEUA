import time
import yaml
import os
import torch
from torch import optim
from tqdm import tqdm
from dataset_generator import get_dataset
from torch.utils.data import DataLoader
from FuNEUA import FuzzyUserAllocator


def experiment():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)[
            'data']
    script_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(script_path)
    user_num = config['user_num']
    x_end = config['x_end']
    y_end = config['y_end']
    min_cov = config['min_cov']
    max_cov = config['max_cov']
    miu = config['miu']
    sigma = config['sigma']
    data_size = config['data_size']
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    user_size = config['user_size']
    n_heads = config['n_heads']
    embed_dim = config['embed_dim']
    device = torch.cuda.current_device()

    dataset = get_dataset(x_end, y_end, miu, sigma, user_num, data_size, min_cov, max_cov, device, dir_name)
    train_loader = DataLoader(dataset=dataset['train'], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset['valid'], batch_size=batch_size, shuffle=False)

    model = FuzzyUserAllocator(user_size, n_heads, embed_dim, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.5)

    best_loss = float('inf')
    best_time = 0

    model_filename = dir_name + "/model/"
    if not os.path.exists(model_filename):
        os.makedirs(model_filename)

    for epoch in range(epochs):
        model.train()
        for _, (tra_server_seq, tra_user_seq, tra_masks) in enumerate(train_loader):
            for batch_idx in tqdm(range(len(tra_server_seq))):
                loss, allocations, server_usage, user_allocated_prop, server_used_prop = model(
                    tra_server_seq[batch_idx],
                    tra_user_seq[batch_idx],
                    tra_masks[batch_idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        print("---------------Evaluate------------------")
        val_loss = 0
        val_user_allocated_prop = 0
        with torch.no_grad():
            for _, (val_server_seq, val_user_seq, val_masks) in enumerate(valid_loader):
                val_server_seq, val_user_seq, val_masks = val_server_seq.to(device), val_user_seq.to(
                    device), val_masks.to(device)
                for val_batch_idx in tqdm(range(len(val_server_seq))):
                    val_loss_batch, _, _, val_user_allocated_prop_batch, _ = model(val_server_seq[val_batch_idx],
                                                                                   val_user_seq[val_batch_idx],
                                                                                   val_masks[val_batch_idx])
                    val_loss += val_loss_batch.item()
                    val_user_allocated_prop += val_user_allocated_prop_batch.item()
            val_loss /= len(valid_loader)
            val_user_allocated_prop /= len(valid_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                best_time = 0
                file_filename = model_filename + time.strftime(
                    '%m%d%H%M', time.localtime(time.time())
                ) + 'best_model.pth'
                torch.save(model.state_dict(), file_filename)
                print(f"Model saved to： {file_filename}")
            else:
                best_time += 1
                print('The model effect has not improved after ' + str(best_time) + ' rounds.')
        scheduler.step(val_loss)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr}')
if __name__ == '__main__':
    experiment()

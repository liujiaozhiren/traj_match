import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
from pathlib import Path
import sys
import cfg
import pickle
import torch.nn.functional as F

from my_diffusion.dyn_diff.model import StepwiseForwardDiffTail

if torch.cuda.is_available():

    # 以当前文件为基准：model.py -> fixLdiff -> my_diffusion -> traj-match(根)
    ROOT = Path('/home/haitaoyuan/data/superJupterNote/traj-match/') # 2 就是两级父目录的上一层：traj-match
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from diffusion.predict.pretrain import valid, pred_mse_loss, loader_collate_fn
from model.handler import ModelHandler
# from my_diffusion.base.model import MyBaseDiff
from my_diffusion.fixLdiff.model import FixLenDiff
from pretrain.code.util import preproc, mean_geo_distance


# def gather(consts: torch.Tensor, t: torch.Tensor):
#     """Gather consts for $t$ and reshape to feature map shape"""
#     c = consts.gather(-1, t)
#     return c.reshape(-1, 1, 1)

def pretrain_Diff_Traj_pred(model, path='pretrain/data/chengdu/chengdu_trajs.pkl'):
    print('cuda available:', torch.cuda.is_available())
    myDiff = FixLenDiff() # StepwiseForwardDiffTail or FixLenDiff

    print('CWD =', os.getcwd())
    abs_path = Path(path).resolve()
    print('Resolved path =', abs_path)
    print('Exists =', abs_path.exists())
    trajs = pickle.load(open(path, "rb"))

    trajs = preproc(trajs, length=50)
    print(f'loading {len(trajs)} trajectories for training the diffusion model...')
    trajs = np.array(trajs)
    print(trajs[0][0])
    trajs[:, :, 0], trajs[:, :, 1] = trajs[:, :, 1], trajs[:, :, 0]
    traj = torch.as_tensor(trajs[:int(len(trajs)*0.99)]).float()
    traj_valid = torch.as_tensor(trajs[int(len(trajs)*0.99):]).float()
    dataset, dataset_valid = TensorDataset(traj), TensorDataset(traj_valid)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    validloader = DataLoader(dataset_valid, batch_size=512, shuffle=False, num_workers=0)

    bst_acc, early_stop = 1e7, 0

    for epoch in range(1, 500 + 1):
        losses, mses = 0, 0
        with tqdm(dataloader, f"train epoch {epoch}") as tq:
            for _, (trainx) in enumerate(tq):
                trainx = trainx[0]
                trainx = torch.transpose(trainx, 1, 2)
                loss, mse = myDiff.learn(trainx)
                losses += loss
                mses += mse
        acc, dist, d2, d4 = gen_valid(myDiff, validloader)
        if acc < bst_acc:
            bst_acc = acc
            early_stop = 0
            torch.save(myDiff.unet.state_dict(), f'pretrain/file/model_para_diff_{acc:.8f}_{cfg.model_name}.pth')
            print(f'saving model with valid mse {bst_acc:.7f} at epoch {epoch}')
        else:
            early_stop += 1
            if early_stop >= cfg.early_stop:
                print('early stopping at epoch', epoch)
                break
        print(f'Epoch {epoch}, Loss: {(losses / len(dataloader)):.7f}, Mse: {(mses / len(dataloader)):.4f}, Valid Mse: {acc:.8f} Valid dist: {dist:.3f}|{d4:.3f}|{d2:.3f} ')


def gen_valid(myDiff, loader):
    myDiff.eval()
    sum_loss, cnt, sum_dist_ = 0.0, 0, 0.0
    sum4,sum2 = 0.0, 0.0
    # sum_dist = 0.0
    with torch.no_grad():
        with tqdm(loader, "valid_") as tq:
            for trajs in tq:
                trajs = trajs[0]
                trajs = torch.transpose(trajs, 1, 2).to(cfg.device)
                info, label = trajs[:, :, :42], trajs[:, :, 42:]
                out = myDiff.infer_from_noise(info)
                loss = F.mse_loss(out, label)
                dist = mean_geo_distance(out, label, reduce='all')
                dist2 = mean_geo_distance(out[...,:2], label[...,:2], reduce='all')
                dist4 = mean_geo_distance(out[...,:4], label[...,:4], reduce='all')
                sum_loss += loss.item()
                sum_dist_ += dist.item()
                sum4 += dist4.item()
                sum2 += dist2.item()

                cnt += 1
    return sum_loss / cnt, sum_dist_ / cnt, sum4 / cnt, sum2 / cnt


def pretrain_vanilla_pred(model, path='pretrain/data/chengdu/chengdu_trajs_small.pkl'):
    assert cfg.input_height == False
    trajs = pickle.load(open(path, "rb"))
    if path.endswith('.pkl'):
        traj_list = []
        for k in trajs.key():
            traj_list.append(trajs[k])
        trajs = traj_list

    train_trajs, valid_trajs = trajs[:int(len(trajs) * 0.8)], trajs[int(len(trajs) * 0.8):]
    validloader = DataLoader(valid_trajs, batch_size=cfg.batch_size, shuffle=False, collate_fn=loader_collate_fn)
    trainloader = DataLoader(train_trajs, batch_size=cfg.batch_size, shuffle=True, collate_fn=loader_collate_fn)

    bst, early_stop = 1e5, 0
    criterion = pred_mse_loss
    model.train()
    model_optim = torch.optim.Adam(model.parameters(), lr=0.00001)

    for epoch in range(50000):
        model.train()
        sum_loss, cnt = 0.0, 0
        with tqdm(trainloader, "train_") as tq:
            for trajs, lens in tq:
                # trajs_in, lens_in = trajs.view(-1, trajs.shape[-2], trajs.shape[-1]), lens.view(-1)
                out = model(trajs)  # vecters [B*SAM, d_model]
                loss = criterion(out)
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()
                sum_loss += loss.item()

        acc, dist = valid(model, validloader, criterion)

        print(
            f'Epoch {epoch + 1}, Loss: {sum_loss / len(trainloader):.7f}, valid_loss: {acc:.6f}|{bst:.6f} dist:{dist:.3f}')

        if acc < bst:
            bst = acc
            early_stop = 0
            bst_pth = f'model_para_{acc}_{cfg.model_name}.pth'
        else:
            early_stop += 1
            if early_stop >= cfg.early_stop:
                break


def pre_train_model(model_name='pred_lstm', path='pretrain/data/chengdu/chengdu_trajs.pkl'):
    model = ModelHandler(dim=cfg.dim, model_name=model_name).to(cfg.device)
    pretrain_vanilla_pred(model, path)


if __name__ == "__main__":
    fn = cfg.train_file
    # fn = 'pretrain/data/chengdu/chengdu_trajs_small.pkl'
    pretrain_Diff_Traj_pred('pred_lstm', fn)
    # model = ModelHandler(dim=cfg.dim, model_name='pred_lstm').to(cfg.device)
    # pretrain_Diff_Traj_pred(model, path='pretrain/data/chengdu/chengdu_trajs.pkl')

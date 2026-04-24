import pickle

import torch, random
from torch.utils.data import DataLoader
from tqdm import tqdm

import cfg
from model.handler import ModelHandler

import torch.nn.functional as F
try:
    from raw_data_proc.db import shenzhen
except ImportError:
    shenzhen = [1746680000000, 114.057868, 22.543099, 500]

def pred_mse_loss(output) -> torch.Tensor:
    pre0, pre1, post0, post1, len_tensor_pre, len_tensor_post = output
    B, L, *_ = pre0.shape
    len_tensor_pre = torch.tensor(len_tensor_pre, device=cfg.device, dtype=torch.long)
    len_tensor_post = torch.tensor(len_tensor_post, device=cfg.device, dtype=torch.long)
    pre_traj_label, post_traj_lable = pre0[:, 1:, :], post0[:, 1:, :]
    pre_traj_out, post_traj_out = pre1[:, :-1, :], post1[:, :-1, :]

    def fill_pad_0(tensor: torch.Tensor, l: torch.Tensor, pred=False) -> torch.Tensor:
        mask = torch.arange(tensor.shape[1], device=cfg.device, dtype=torch.float64).unsqueeze(0) < (l-1).unsqueeze(1)
        mask[:, 0:cfg.infer_start] = False  # 前面不参与计算
        return tensor * mask.unsqueeze(-1)

    pad_pre_traj_label, pad_post_traj_lable = fill_pad_0(pre_traj_label, len_tensor_pre), fill_pad_0(post_traj_lable, len_tensor_post)
    pad_pre_traj_out, pad_post_traj_out = fill_pad_0(pre_traj_out, len_tensor_pre, pred=True), fill_pad_0(post_traj_out, len_tensor_post, pred=True)

    pre_loss = F.mse_loss(pad_pre_traj_label[..., 1:], pad_pre_traj_out[..., 1:], reduction='none').sum(dim=-1).mean()
    post_loss = F.mse_loss(pad_post_traj_lable[..., 1:], pad_post_traj_out[..., 1:], reduction='none').sum(dim=-1).mean()
    loss = (pre_loss + post_loss) / 2.0
    return loss

def haversine_dist(label: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    R = 6_371_000.0                                          # 地球半径（米）
    lon1, lat1 = torch.deg2rad(label[..., 0]), torch.deg2rad(label[..., 1])
    lon2, lat2 = torch.deg2rad(pred[..., 0]), torch.deg2rad(pred[..., 1])
    dlon, dlat = lon2 - lon1, lat2 - lat1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.arcsin(torch.clamp(torch.sqrt(a), max=1.0))  # 数值安全
    return R * c

def pre_dist_loss(output) -> torch.Tensor:
    assert cfg.input_height == False, "仅支持经纬度输入"
    assert output[0].shape[-1] == 3, "输入至少包含时间戳、经度、纬度"
    pre0, pre1, post0, post1, len_tensor_pre, len_tensor_post = output
    B, L, *_ = pre0.shape
    len_tensor_pre = torch.tensor(len_tensor_pre, device=cfg.device, dtype=torch.long)
    len_tensor_post = torch.tensor(len_tensor_post, device=cfg.device, dtype=torch.long)
    pre_traj_label, post_traj_lable = pre0[:, 1:, :], post0[:, 1:, :]
    pre_traj_out, post_traj_out = pre1[:, :-1, :], post1[:, :-1, :]

    def fill_pad_0(tensor: torch.Tensor, l: torch.Tensor, pred=False) -> torch.Tensor:
        mask = torch.arange(tensor.shape[1], device=cfg.device, dtype=torch.float64).unsqueeze(0) < (l-1).unsqueeze(1)
        mask[:, 0:cfg.infer_start] = False  # 前面不参与计算
        return tensor * mask.unsqueeze(-1)

    pad_pre_label, pad_post_label = fill_pad_0(pre_traj_label, len_tensor_pre), fill_pad_0(post_traj_lable, len_tensor_post)
    pad_pre_out, pad_post_out = fill_pad_0(pre_traj_out, len_tensor_pre, pred=True), fill_pad_0(post_traj_out, len_tensor_post, pred=True)

    pre_loss = haversine_dist(pad_pre_label[..., 1:], pad_pre_out[..., 1:]).mean()
    post_loss = haversine_dist(pad_post_label[..., 1:], pad_post_out[..., 1:]).mean()

    return (pre_loss + post_loss) / 2.0


def split_list_with_controlled_overlap(traj, k, a, b, max_overlap_ratio=0.3, max_trials=1000):

    L = len(traj)
    result = []
    used_ranges = []

    def is_valid(start, end):
        for s, e in used_ranges:
            overlap = max(0, min(end, e) - max(start, s))
            allowed = int(min(end - start, e - s) * max_overlap_ratio)
            if overlap > allowed:
                return False
        return True

    trials = 0
    while len(result) < k and trials < max_trials:
        length = random.randint(a, b)
        if length > L:
            break
        start = random.randint(0, L - length)
        end = start + length
        if is_valid(start, end):
            result.append(traj[start:end])
            used_ranges.append((start, end))
        trials += 1

    return result

def get_mid_point(traj):
    lon, lat, timestamp = [], [], []
    for point in traj:
        t, lon_, lat_ = point[0], point[1], point[2]
        lon.append(lon_)
        lat.append(lat_)
        timestamp.append(t)
    mid_lon = sum(lon) / len(lon)
    mid_lat = sum(lat) / len(lat)
    mid_timestamp = int(sum(timestamp) / len(timestamp))
    return mid_timestamp, mid_lon, mid_lat


def clean_traj(traj):
    mid_t, mid_lon, mid_lat = get_mid_point(traj[-10:])
    new_traj = []
    for point in traj:
        t, lon, lat = point[0], point[1], point[2]
        dt, dlon, dlat = t-mid_t, lon-mid_lon, lat-mid_lat
        dt = (dt//3) * 1000
        new_traj.append([dt + shenzhen[0], dlat + shenzhen[1], dlon + shenzhen[2]])
    return new_traj

def clean_trajs(trajs):
    for i in range(len(trajs)):
        trajs[i] = clean_traj(trajs[i])


def loader_collate_fn(batch):
    batch_size = len(batch)
    pre_traj, post_traj = [],[]
    for traj in batch:
        if len(traj) <= 30:
            continue
        pre_, post_ = traj, traj[::-1]
        pre_s = split_list_with_controlled_overlap(pre_, len(traj)//40, 30, 50)
        post_s = split_list_with_controlled_overlap(post_, len(traj)//40, 30, 50)
        clean_trajs(pre_s)
        clean_trajs(post_s)
        pre_traj.extend(pre_s)
        post_traj.extend(post_s)
    d = len(pre_traj[0][0])
    lengths_pre = torch.tensor([len(traj) for traj in pre_traj], dtype=torch.int32, device=cfg.device)
    lengths_post = torch.tensor([len(traj) for traj in post_traj], dtype=torch.int32, device=cfg.device)
    max_len_pre, max_len_post = int(lengths_pre.max()), int(lengths_post.max())
    max_l = max(max_len_pre, max_len_post) + cfg.max_predict
    padded_pre = torch.zeros(len(pre_traj), max_l, d, dtype=torch.float64, device=cfg.device)
    padded_post = torch.zeros(len(post_traj), max_l, d, dtype=torch.float64, device=cfg.device)
    for i, traj in enumerate(pre_traj):
        L = len(traj)
        padded_pre[i, :L, :] = torch.tensor(traj, dtype=torch.float64, device=cfg.device)
        #padded_pre[i, :, 0] = torch.tensor(complete_timestamp(max_l, traj), dtype=torch.float64, device=cfg.device)
    for i, traj in enumerate(post_traj):
        L = len(traj)
        padded_post[i, :L, :] = torch.tensor(traj, dtype=torch.float64, device=cfg.device)
        #padded_post[i, :, 0] = torch.tensor(complete_timestamp(max_l, traj[::-1]), dtype=torch.float64, device=cfg.device)
    ret = ((padded_pre, padded_post), (lengths_pre, lengths_post)), None
    return ret

def valid(model, loader, criterion):
    with torch.no_grad():
        model.eval()
        sum_loss= 0
        sum_dist = 0
        for trajs, lens in loader:
            #trajs_in, lens_in = trajs.view(-1, trajs.shape[-2], trajs.shape[-1]), lens.view(-1)
            out = model(trajs)
            loss = criterion(out)
            pre0, pre1, post0, post1,l1,l2 = out
            pre_input = model.gen_pre.norm.unnorm(pre0)
            post_input = model.gen_post.norm.unnorm(post0)
            pre_output = model.gen_pre.norm.unnorm(pre1)
            post_output = model.gen_post.norm.unnorm(post1)
            tmp_input = pre_input, pre_output, post_input,post_output,l1, l2
            dist_loss = pre_dist_loss(tmp_input)
            sum_loss += loss.item()
            sum_dist += dist_loss.item()
    return sum_loss / len(loader), sum_dist / len(loader)

def pretrain_lstm_pred(model, path='pretrain/data/chengdu/chengdu_trajs.pkl'):
    assert cfg.input_height == False
    trajs = pickle.load(open(path, "rb"))
    if path.endswith('.pkl'):
        traj_list =[]
        for k in trajs.key():
            traj_list.append(trajs[k])
        trajs = traj_list

    train_trajs, valid_trajs = trajs[:int(len(trajs) * 0.8)], trajs[int(len(trajs) * 0.8):]
    validloader = DataLoader(valid_trajs, batch_size=cfg.batch_size, shuffle=False, collate_fn=loader_collate_fn)
    trainloader = DataLoader(train_trajs, batch_size=cfg.batch_size, shuffle=True, collate_fn=loader_collate_fn)
    # model, train_loader, validloader, testloader = preproc_all(model_name, train_set, valid_set, db, all)
    # valid_func = valid_func_dict[model_name]

    bst, early_stop = 1e5, 0
    criterion = pred_mse_loss
    model.train()
    model_optim = torch.optim.Adam(model.parameters(), lr=0.00001)

    # acc = valid_func(validloader, model, -1)
    # print(f'acc start {acc}')
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
        # if epoch % 2 != 0:
        #     print(f'Epoch {epoch + 1}, Loss: {sum_loss / len(train_loader):.4f}')
        #     continue

        acc,dist = valid(model, validloader, criterion)
        #t_acc = valid(testloader, model, epoch)
        print(
            f'Epoch {epoch + 1}, Loss: {sum_loss / len(trainloader):.7f}, valid_loss: {acc:.6f}|{bst:.6f} dist:{dist:.3f}')
        #bt = max(t_acc, bt)
        if acc < bst:
            bst = acc
            early_stop = 0
            #torch.save(model.state_dict(), f'model_para_{acc}_{cfg.model_name}.pth')
            bst_pth = f'model_para_{acc}_{cfg.model_name}.pth'
            # print(f'Best acc:{acc:.4f}  save model_para_{acc}_{cfg.model_name}.pth')
        else:
            early_stop += 1
            if early_stop >= cfg.early_stop:
                break
            # print(f'worse acc:{acc:.4f}')
        # print(f'HR5:{bst5:.4f},HR10:{bst10:.4f}, HR50:{bst50:.4f}, NDCG:{bndcg:.7f}, HR10in50:{bst10in50:.4f}')

    # model.load_state_dict(torch.load(bst_pth, map_location=cfg.device))

if __name__ == '__main__':
    #pretrain_lstm_pred(model=None, path='./tmp')
    #trajs = [[[0,1,1],[1,2,2,],[2,3,3],[3,4,4],[4,5,5]],[[0,1,1],[1,2,2,],[2,3,3],[3,4,4],[4,5,5]],[[0,1,1],[1,2,2,],[2,3,3],[3,4,4],[4,5,5]]]
    model = ModelHandler(dim=cfg.dim, model_name="pred_lstm").to(cfg.device)
    pretrain_lstm_pred(model, 'tmp')

    #trajs = pickle.load(open('tmp', "rb"))
    #loader_collate_fn(trajs)
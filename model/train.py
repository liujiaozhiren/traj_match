import torch.cuda
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import cfg
from diffusion.predict.loader import pred_lstm_preproc
from diffusion.util import sample_dist_cal
from match_model.my.match_train import mydiff_preproc
from model.ST2Vec.preproc import ST2Vec_preproc
from model.handler import ModelHandler
from model.traj2simvec.pre_proc import traj2SimVec_preproc
from model.trajGAT.pre_proc import trajGAT_preproc



def triplet_loss_batch_hard(x: torch.Tensor, margin: float = 0.2, normalize: bool = True):
    """
    a: (B, D), b: (B, D). 期望 a[i] 与 b[i] 为正对，其余为负。
    使用 batch-hard：对每个 i，正样本距离 = d(a[i], b[i])，
    负样本距离取 min_j d(a[i], b[j]) (j!=i) 作为 hardest negative。
    """
    a = x[0]
    b = x[1]
    if normalize:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)

    # 余弦距离/欧式距离都可，这里用余弦距离：d = 1 - cos
    sim = a @ b.t()                      # (B, B), 越大越相似
    pos = sim.diag()                     # (B,)
    # hardest negative: 选非对角的最大相似度（因为相似度越大越“难”）
    B = sim.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    neg_max = sim.masked_fill(~mask, float('-inf')).max(dim=1).values  # (B,)

    # 把相似度转为“距离”再做 hinge：d = 1 - sim
    pos_d = 1 - pos
    neg_d = 1 - neg_max
    loss = F.relu(pos_d - neg_d + margin).mean()
    return loss

def info_nce_cross_group(
        x: torch.Tensor,
        temperature: float = 0.07,
        normalize: bool = True
) -> torch.Tensor:
    assert x.dim() == 3 and x.size(0) == 2, "shape must be (2, B, D)"
    g0, g1 = x[0], x[1]  # (B, D) each
    if normalize:
        g0, g1 = F.normalize(g0, dim=-1), F.normalize(g1, dim=-1)

    logits = g0 @ g1.T  # (B, B)
    logits = logits / temperature

    labels = torch.arange(g0.size(0), device=x.device)
    loss = F.cross_entropy(logits, labels, reduction="mean")
    return loss

def info_nce_bidir(
    x: torch.Tensor,
    temperature: float = 0.07,
    normalize: bool = True
) -> torch.Tensor:
    """
    x: (2, B, D) -> z0, z1
    """
    assert x.dim() == 3 and x.size(0) == 2, "shape must be (2, B, D)"
    z0, z1 = x[0], x[1]             # (B, D)
    if normalize:
        z0 = F.normalize(z0, dim=-1)
        z1 = F.normalize(z1, dim=-1)

    logits01 = (z0 @ z1.T) / temperature  # (B, B)
    logits10 = (z1 @ z0.T) / temperature  # (B, B)
    labels = torch.arange(z0.size(0), device=x.device)

    loss01 = F.cross_entropy(logits01, labels, reduction="mean")
    loss10 = F.cross_entropy(logits10, labels, reduction="mean")
    return 0.5 * (loss01 + loss10)


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

loss_dict = {
    '': info_nce_cross_group,
    'traj2simvec': info_nce_bidir,
    'trajGAT': info_nce_bidir,
    'ST2Vec': info_nce_bidir,
    'pred_lstm': pred_mse_loss,
    'mydiff':None
}




def valid(valid_loader, model, epoch):
    model.eval()
    traj_embed_list = torch.zeros((2, len(valid_loader.dataset), cfg.dim), dtype=torch.float32, device=cfg.device)
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            trajs, lens = batch
            embed = model(trajs)
            traj_embed_list[0, i * cfg.batch_size:(i + 1) * cfg.batch_size] = embed[0]
            traj_embed_list[1, i * cfg.batch_size:(i + 1) * cfg.batch_size] = embed[1]
            # traj_embed_list.extend(embed.squeeze())
    return accuracy(traj_embed_list, normalize=True)

def pred_valid(valid_loader, model, epoch):
    traj_dup_list_pre, traj_dup_list_post = [], []
    with torch.no_grad():
        with tqdm(valid_loader, desc=f"Valid Epoch {epoch}") as tq:
            for i, batch in enumerate(tq):
                trajs, lens = batch
                trajs, lens = trajs
                t = cfg.max_predict
                input = trajs, lens, t
                output = model.gen_trajs(input)
                traj_dup_list_pre.extend(output[0])
                traj_dup_list_post.extend(output[1])
    return pred_accuracy(traj_dup_list_pre, traj_dup_list_post, cfg.max_predict)

def accuracy(
        x: torch.Tensor,
        normalize: bool = True
) -> torch.Tensor:
    assert x.dim() == 3 and x.size(0) == 2, "shape must be (2, B, D)"
    g0, g1 = x[0], x[1]  # (B, D) anchors  &  candidates

    if normalize:
        g0, g1 = F.normalize(g0, dim=-1), F.normalize(g1, dim=-1)

    logits = g0 @ g1.T  # (B, B)
    pred = logits.argmax(dim=1)  # 取列索引
    labels = torch.arange(g0.size(0), device=x.device)
    acc = (pred == labels).float().mean()  # 标量 Tensor
    return acc

def pred_accuracy(x_pre, x_post, t):
    dist_matrix = torch.zeros((len(x_pre), len(x_post)), dtype=torch.float32, device=cfg.device)
    with tqdm(x_pre, desc="Calculating octopus Distances") as tq:
        for i, pre in enumerate(tq):
            for j, post in enumerate(x_post):
                dist_matrix[i, j] = sample_dist_cal(pre, post, t)

    if cfg.assignment_method == 'hungarian':
        from scipy.optimize import linear_sum_assignment  # pip install scipy
        row_idx, col_idx = linear_sum_assignment(dist_matrix.cpu().numpy())
        pred = torch.as_tensor(col_idx, device=cfg.device)
    else:
        pred = dist_matrix.argmin(dim=1)  # 取列索引
    labels = torch.arange(len(pred), device=cfg.device)
    acc = (pred == labels).float().mean()  # 标量 Tensor
    return acc


def preproc_all(model_name, train_set=None, valid_set=None, db=None, all=None):
    model, train_loader, validloader, testloader = None, None, None, None
    if model_name == 'traj2simvec':
        model, train_loader, validloader, testloader = traj2SimVec_preproc(model_name, train_set, valid_set, db)
    elif model_name == 'trajGAT':
        model, train_loader, validloader, testloader = trajGAT_preproc(train_set, valid_set, db, all)
    elif model_name == 'ST2Vec':
        model, train_loader, validloader, testloader = ST2Vec_preproc(train_set, valid_set, db, all)
    elif model_name == 'pred_lstm':
        model, train_loader, validloader, testloader = pred_lstm_preproc(model_name, train_set, valid_set, db, all)
    elif model_name == 'mydiff':
        model, train_loader, validloader, testloader = mydiff_preproc(model_name, train_set, valid_set, db, all)
    return model, train_loader, validloader, testloader


valid_func_dict = {
    '': valid,
    'traj2simvec': valid,
    'trajGAT': valid,
    'ST2Vec': valid,
    'pred_lstm': pred_valid,
    'mydiff':None
}

def model_train(train_set_, db, all, model_name='traj2simvec'):
    # import gc
    # gc.collect()
    train_set = train_set_[:int(len(train_set_) * cfg.train_ratio)]
    valid_set = train_set_[int(len(train_set_) * cfg.train_ratio):]

    model, train_loader, validloader, testloader = preproc_all(model_name, train_set, valid_set, db, all)
    valid_func = valid_func_dict[model_name]
    bst, bt, early_stop = 0.0, 0.0, 0
    criterion = loss_dict[model_name]
    model.train()
    model_optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # acc = valid_func(validloader, model, -1)
    # print(f'acc start {acc}')
    for epoch in range(50000):
        model.train()
        sum_loss, cnt = 0.0, 0
        with tqdm(train_loader, "train_") as tq:
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

        acc = valid_func(validloader, model, epoch)
        t_acc = valid_func(testloader, model, epoch)
        print(
            f'Epoch {epoch + 1}, Loss: {sum_loss / len(train_loader):.7f}, Acc: {acc:.4f}|{bst:.4f}, tacc:{t_acc:.4f}|{bt:.4f}')
        bt = max(t_acc, bt)
        if acc >= bst:
            bst = acc
            early_stop = 0
            torch.save(model.state_dict(), f'model_para_{acc}_{cfg.model_name}.pth')
            bst_pth = f'model_para_{acc}_{cfg.model_name}.pth'
            # print(f'Best acc:{acc:.4f}  save model_para_{acc}_{cfg.model_name}.pth')
        else:
            early_stop += 1
            if early_stop >= cfg.early_stop:
                break
            # print(f'worse acc:{acc:.4f}')
        # print(f'HR5:{bst5:.4f},HR10:{bst10:.4f}, HR50:{bst50:.4f}, NDCG:{bndcg:.7f}, HR10in50:{bst10in50:.4f}')

    model.load_state_dict(torch.load(bst_pth, map_location=cfg.device))

    return valid_func(testloader, model, -1)

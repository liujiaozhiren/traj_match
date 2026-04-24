import numpy as np
import torch
from tqdm import tqdm
import cfg
from pipeline.diff import gen_preproced_loader
from pipeline.downstream import downstream_fn
from pipeline.load import hydra_graph
from pipeline.rl_selector import RLConfig, RLSelector


def make_downstream_fn(get_hydra_graph_label, match, loss_func):
    def fn(info_pre, info_post, sel_pre, sel_post, label):
        bg_pairs, _ = get_hydra_graph_label(info_pre, info_post, sel_pre, sel_post, pair=cfg.pair_match)
        ret = match(bg_pairs, pairs=cfg.pair_match)
        # 要求返回 per-sample loss：请在你的 loss 实现里支持 reduction='none'
        # print('sjn', ret.shape, label.shape)
        loss_vec = loss_func(ret, label)  # (B,)
        B = loss_vec.shape[0] // 2
        loss = loss_vec.reshape(2,B).mean(dim=0)  # (B,)
        return loss
    return fn





def rl2_train(train_db, db, preDiff=None, postDiff=None, match=None):
    # 数据：你已有

    train_loader, valid_loader, valid_p = gen_preproced_loader(preDiff, postDiff, train_db, db)
    valid_info_post, valid_hydra_post = valid_p


    # RL模块：一键切换算法
    rl_cfg = RLConfig(algo=getattr(cfg, "rl_algo", "ppo"),
                      k_select=cfg.k_select,
                      hidden=cfg.dim,
                      ent_coef=1e-3, vf_coef=0.5,
                      clip_eps=0.2, ppo_epochs=3)
    rl = RLSelector(d_info=2, d_cand=2, cfg=rl_cfg).to(cfg.device)
    rl_optim = torch.optim.AdamW(rl.parameters(), lr=3e-4, weight_decay=1e-4)

    # loss_func = pair_loss if cfg.pair_match else embedding_loss


    for epoch in range(1, 500):
        rl.train()
        with tqdm(train_loader, desc=f"RL epoch {epoch}") as tq:
            for info_pre, info_post, hydra_pre, hydra_post in tq:
                info_pre   = info_pre.to(cfg.device)
                info_post  = info_post.to(cfg.device)
                hydra_pre  = hydra_pre.to(cfg.device)
                hydra_post = hydra_post.to(cfg.device)



                B = info_pre.shape[0]

                perm = torch.roll(torch.arange(B, device=cfg.device), shifts=1, dims=0)  # [1,2,...,B-1,0]
                info_post_neg = info_post[perm]  # (B, L, D)
                hydra_post_neg = hydra_post[perm]  # (B, T, L, D)

                # --- 拼接正负样本到同一批次 ---
                info_pre_all = torch.cat([info_pre, info_pre], dim=0)  # (2B, L, D)
                hydra_pre_all = torch.cat([hydra_pre, hydra_pre], dim=0)  # (2B, T, L, D)
                info_post_all = torch.cat([info_post, info_post_neg], dim=0)  # (2B, L, D)
                hydra_post_all = torch.cat([hydra_post, hydra_post_neg], dim=0)  # (2B, T, L, D)

                y_pos = torch.ones(B, dtype=torch.long, device=cfg.device)
                y_neg = torch.zeros(B, dtype=torch.long, device=cfg.device)
                y_all = torch.cat([y_pos, y_neg], dim=0)

                logs = rl.step(info_pre_all, info_post_all, hydra_pre_all, hydra_post_all, match, rl_optim, y_all)
                tq.set_postfix(rl=f"{logs['rl_loss']:.4f}", down=f"{logs['down_loss_mean']:.4f}")

        # # ===== 验证（贪心选择）=====
        rl.eval()
        with torch.no_grad():
            num = 0
            N_all = len(valid_info_post)
            all_mat = torch.zeros(N_all, N_all, device=cfg.device)
            base_mat = torch.zeros(N_all, N_all, device=cfg.device)
            v_info_post = valid_info_post.to(cfg.device)
            v_hydra_post = valid_hydra_post.to(cfg.device)
            with tqdm(valid_loader, desc=f"RL valid epoch {epoch}") as tq:
                for v_info_pre, _, v_hydra_pre, _ in tq:
                    v_info_pre   = v_info_pre.to(cfg.device) # Batch x len x dim
                    v_hydra_pre  = v_hydra_pre.to(cfg.device) # Batch x hydra_tail x len x dim
                    Bv, L, D = v_info_pre.shape
                    Bv, T, L2, D = v_hydra_pre.shape

                    pre_tile = v_info_pre[:,None, ...].expand(Bv, N_all, L,D).reshape(Bv*N_all, L,D)
                    post_tile = v_info_post[None, ...].expand(Bv, N_all, L,D).reshape(Bv*N_all, L,D)

                    hydra_pre_tile = v_hydra_pre[:,None,...].expand(Bv, N_all, T, L2,D).reshape(Bv*N_all, T, L2,D)
                    hydra_post_tile = v_hydra_post[None,...].expand(Bv, N_all, T, L2,D).reshape(Bv*N_all, T, L2,D)

                    # ===== 贪心选择 + 匹配 =====
                    sel_pre, sel_post = rl.choose(pre_tile, post_tile,
                                                  hydra_pre_tile, hydra_post_tile,
                                                  greedy=True)

                    # bg_pairs = hydra_graph(pre_tile, post_tile, sel_pre, sel_post, pair=cfg.pair_match)
                    # ret = match(bg_pairs, pairs=cfg.pair_match)  # (Bv*N_all,) 或 (Bv*N_all,1)
                    ret = downstream_fn(match, pre_tile, post_tile, sel_pre, sel_post, label=None)

                    all_mat[num:num + Bv] = ret.view(Bv, N_all)

                    no_sel_pre = hydra_pre_tile[:, :cfg.k_select, ...]  # (Bv*N_all, 4, L, D)
                    no_sel_post = hydra_post_tile[:, :cfg.k_select, ...]  # (Bv*N_all, 4, L, D)

                    bg_pairs_b = hydra_graph(pre_tile, post_tile, no_sel_pre, no_sel_post, pair=cfg.pair_match)
                    ret_b = match(bg_pairs_b, pairs=cfg.pair_match)

                    base_mat[num:num + Bv] = ret_b.view(Bv, N_all)

                    num += Bv

            acc_list, base_acc = [], []
            for i in range(len(valid_info_post)):
                row = all_mat[i]
                acc_list.append(int(row[i] == torch.max(row)))
                base_row = base_mat[i]
                base_acc.append(int(base_row[i] == torch.max(base_row)))
            e_acc = float(torch.tensor(acc_list, device=cfg.device).float().mean().item())
            e_base_acc = float(torch.tensor(base_acc, device=cfg.device).float().mean().item())
            print(f"{epoch} E [Valid] RL Acc: {e_acc:.3f}, Base Acc: {e_base_acc:.3f}")



#
# def rl_train(train_db, db, preDiff=None, postDiff=None, match=None):
#     action_low = None
#     action_high = None
#     rl = RLPolicy(dim_in=2, act_dim=cfg.dim*4*2, d_model=256, hidden=256,
#                   use_transformer=True, action_low=action_low, action_high=action_high).to(cfg.device)
#     rl_optim = torch.optim.AdamW(rl.parameters(), lr=3e-4, weight_decay=1e-4)
#     bst_acc, bst_dist, early_stop = 0, 0, 0
#     loss_func = pair_loss if cfg.pair_match else embedding_loss
#     train_loader, valid_loader, valid_p = gen_preproced_loader(preDiff, postDiff, train_db, db)
#     valid_info_post, valid_hydra_post = valid_p
#
#
#     for epoch in range(1, 500 + 1):
#         loss_list, acc_list, acc_base_list = [], [], []
#         with tqdm(train_loader, f"RL epoch {epoch}") as tq:
#             for info_pre, info_post, hydra_pre, hydra_post in tq:
#
#
#                 # ---- RL 前向(需要梯度) ----
#                 rl_in = torch.concat([info_pre, info_post], dim=1)
#                 rl_in = [info_pre, info_post]
#                 chose_in = [hydra_pre, hydra_post]
#
#                 action, logp, value = rl_chose_hydra(rl, rl_in, chose_in)
#                 hydra_pre, hydra_post = action
#
#                 bg_pairs, label = get_hydra_graph_label(info_pre, info_post, hydra_pre, hydra_post, pair=cfg.pair_match)
#                 ret = match(bg_pairs, pairs=cfg.pair_match)
#                 loss_act = loss_func(ret, label, dim=cfg.match_dim)
#
#
#
#                 action, logp, value = get_rl_action(rl, rl_in)  # logp:[B], value:[B] 仍然保留梯度
#
#                 # ---- 环境前向 & 奖励度量(不需要梯度) ----
#                 with torch.no_grad():
#                     # 动作版
#                     hydra_pre = torch.zeros(info_pre.shape[0], cfg.hydra_tail, info_pre.shape[1], cfg.diff_infer_len,
#                                             device=cfg.device)
#                     hydra_post = torch.zeros(info_post.shape[0], cfg.hydra_tail, info_post.shape[1], cfg.diff_infer_len,
#                                              device=cfg.device)
#                     for i in range(cfg.hydra_tail):
#                         out_pre = preDiff.infer_from_noise(info_pre, action[..., :cfg.dim * 4])
#                         out_post = postDiff.infer_from_noise(info_post, action[..., cfg.dim * 4:])
#                         hydra_pre[:, i] = out_pre
#                         hydra_post[:, i] = out_post
#                     # bg_pairs, label = get_hydra_graph_label(
#                     #     info_pre, info_post.flip(dims=[-1]),
#                     #     hydra_pre, hydra_post.flip(dims=[-1]),
#                     #     pair=cfg.pair_match
#                     # )
#                     bg_pairs, label = get_hydra_graph_label(info_pre, info_post,hydra_pre, hydra_post,pair=cfg.pair_match)
#
#                     ret = match(bg_pairs, pairs=cfg.pair_match)
#                     loss_act = loss_func(ret, label, dim=cfg.match_dim)
#
#                     # 基线（action=None）
#                     hydra_pre_b = torch.zeros_like(hydra_pre)
#                     hydra_post_b = torch.zeros_like(hydra_post)
#                     for i in range(cfg.hydra_tail):
#                         out_pre_b = preDiff.infer_from_noise(info_pre, action=None)
#                         out_post_b = postDiff.infer_from_noise(info_post, action=None)
#                         hydra_pre_b[:, i] = out_pre_b
#                         hydra_post_b[:, i] = out_post_b
#                     # bg_pairs_b, _ = get_hydra_graph_label(
#                     #     info_pre, info_post.flip(dims=[-1]),
#                     #     hydra_pre_b, hydra_post_b.flip(dims=[-1]),
#                     #     pair=cfg.pair_match
#                     # )
#                     bg_pairs_b, _ = get_hydra_graph_label(info_pre, info_post,hydra_pre_b, hydra_post_b,pair=cfg.pair_match)
#                     ret_b = match(bg_pairs_b, pairs=cfg.pair_match)
#                     loss_base = loss_func(ret_b, label, dim=cfg.match_dim)
#
#                     # 奖励：基线->动作的改进 - 动作代价
#                     r_improve = (loss_base - loss_act)
#                     act_cost = getattr(cfg, "lambda_action", 1e-3) * action.pow(2).mean()
#                     reward = (r_improve - act_cost).detach()  # 标量或可做 batch 平均
#
#                 # ---- 仅优化 RL ----
#                 adv = reward - value.detach().mean()  # 简洁 baseline
#                 policy_loss = -(logp.mean() * adv)
#                 value_loss = F.mse_loss(value, torch.full_like(value, reward))
#                 entropy_term = rl.dist_entropy() if hasattr(rl, "dist_entropy") else torch.tensor(0.0,
#                                                                                                   device=value.device)
#                 rl_loss = policy_loss + getattr(cfg, "value_coef", 0.5) * value_loss - getattr(cfg, "entropy_coef",
#                                                                                                1e-3) * entropy_term
#
#                 rl_optim.zero_grad()
#                 rl_loss.backward()
#                 rl_optim.step()
#
#                 loss_list.append(loss_act.item())  # 记录动作版监督度量(仅作日志)
#         # valid
#         with tqdm(validloader, f"RL valid epoch {epoch}") as tq:
#             for trajs in tq:
#                 traj_pre = trajs[:, 0, :, 1:]
#                 traj_post = trajs[:, 1, :, 1:]
#                 info_pre = torch.transpose(traj_pre, 1, 2)
#                 info_post = torch.transpose(traj_post, 1, 2)
#
#                 B, C, _ = info_pre.shape
#                 hydra_pre_rl = torch.zeros(B, cfg.hydra_tail, C, cfg.diff_infer_len, device=cfg.device)
#                 hydra_post_rl = torch.zeros(B, cfg.hydra_tail, C, cfg.diff_infer_len, device=cfg.device)
#                 hydra_pre_base = torch.zeros(B, cfg.hydra_tail, C, cfg.diff_infer_len, device=cfg.device)
#                 hydra_post_base = torch.zeros(B, cfg.hydra_tail, C, cfg.diff_infer_len, device=cfg.device)
#
#                 with torch.no_grad():
#                     # 1) RL 确定性动作
#                     rl_in = torch.concat([info_pre, info_post], dim=1)
#                     action_eval, _, _ = get_rl_action(rl, rl_in, eval_mode=True)  # [B, action_dim]
#
#                     # 2) 生成 hydra：RL & Baseline
#                     for i in range(cfg.hydra_tail):
#                         # RL
#                         out_pre_rl = preDiff.infer_from_noise(info_pre, action_eval[..., :cfg.dim * 4])
#                         out_post_rl = postDiff.infer_from_noise(info_post, action_eval[..., cfg.dim * 4:])
#                         hydra_pre_rl[:, i] = out_pre_rl
#                         hydra_post_rl[:, i] = out_post_rl
#                         # Baseline (action=None)
#                         out_pre_b = preDiff.infer_from_noise(info_pre)
#                         out_post_b = postDiff.infer_from_noise(info_post)
#                         hydra_pre_base[:, i] = out_pre_b
#                         hydra_post_base[:, i] = out_post_b
#
#                     # 3) 图配与指标：RL
#                     # bg_pairs_rl, label = get_hydra_graph_label(
#                     #     info_pre, info_post.flip(dims=[-1]),
#                     #     hydra_pre_rl, hydra_post_rl.flip(dims=[-1]),
#                     #     pair=cfg.pair_match, valid=True
#                     # )
#                     bg_pairs_rl, label = get_hydra_graph_label(info_pre, info_post,hydra_pre_rl, hydra_post_rl,pair=cfg.pair_match, valid=True)
#                     ret_rl = match(bg_pairs_rl, pairs=cfg.pair_match)  # [B]
#                     assert cfg.pair_match
#                     acc_rl = (ret_rl > 0).float().tolist()
#                     acc_list.extend(acc_rl)
#
#                     # 4) 图配与指标：Baseline（可选对比）
#                     # bg_pairs_b, _ = get_hydra_graph_label(
#                     #     info_pre, info_post.flip(dims=[-1]),
#                     #     hydra_pre_base, hydra_post_base.flip(dims=[-1]),
#                     #     pair=cfg.pair_match, valid=True
#                     # )
#                     bg_pairs_b, _ = get_hydra_graph_label(info_pre, info_post,hydra_pre_base, hydra_post_base,pair=cfg.pair_match, valid=True)
#                     ret_b = match(bg_pairs_b, pairs=cfg.pair_match)  # [B]
#                     acc_b = (ret_b > 0).float().tolist()
#                     acc_base_list.extend(acc_b)
#
#         e_acc = np.mean(acc_list)
#         e_acc_base = np.mean(acc_base_list) if len(acc_base_list) > 0 else float('nan')
#         print(f"[Valid] RL Acc: {e_acc:.3f} | Base Acc: {e_acc_base:.3f}")
#
#         if e_acc >= bst_acc:
#             bst_acc = e_acc
#             early_stop = 0
#             torch.save(match.state_dict(), f'model_match_{e_acc}_{cfg.model_name}.pth')
#             bst_pth = f'model_match_{e_acc}_{cfg.model_name}.pth'
#             print(f'Best acc:{e_acc:.4f}  save model_para_{e_acc}_{cfg.model_name}.pth')
#         else:
#             early_stop += 1
#             if early_stop >= cfg.early_stop:
#                 break
#             # print(f'worse acc:{acc:.4f}')
#         # print(f'HR5:{bst5:.4f},HR10:{bst10:.4f}, HR50:{bst50:.4f}, NDCG:{bndcg:.7f}, HR10in50:{bst10in50:.4f}')
#
#     # model.load_state_dict(torch.load(bst_pth, map_location=cfg.device))
#
#     return



#
#
# def rl2_train(train_db, db, preDiff=None, postDiff=None, match=None):
#     action_low = None
#     action_high = None
#     rl = RLPolicy(...).to(cfg.device)
#     rl_optim = torch.optim.AdamW(rl.parameters(), lr=3e-4, weight_decay=1e-4)
#     bst_acc, bst_dist, early_stop = 0, 0, 0
#     loss_func = pair_loss if cfg.pair_match else embedding_loss
#     train_loader, valid_loader, valid_p = gen_preproced_loader(preDiff, postDiff, train_db, db)
#     valid_info_post, valid_hydra_post = valid_p
#
#
#     for epoch in range(1, 500 + 1):
#         loss_list, acc_list = [], []
#         with tqdm(train_loader, f"RL epoch {epoch}") as tq:
#             for info_pre, info_post, hydra_pre, hydra_post in tq:
#                 rl_in = [info_pre, info_post]
#                 chose_in = [hydra_pre, hydra_post]
#
#                 for ...:
#                     action, logp, value = rl_chose_hydra(rl, rl_in, chose_in, ....)
#                     hydra_pre, hydra_post = action
#                     bg_pairs, label = get_hydra_graph_label(info_pre, info_post, hydra_pre, hydra_post, pair=cfg.pair_match)
#                     ret = match(bg_pairs, pairs=cfg.pair_match)
#                     loss = loss_func(ret, label, dim=cfg.match_dim)
#                 reward = ...
#                 rl_loss = ...
#                 rl_optim.zero_grad()
#                 rl_loss.backward()
#                 rl_optim.step()
#
#         # valid
#         with tqdm(valid_loader, f"RL valid epoch {epoch}") as tq:
#             num = 0
#             all = torch.zeros(len(valid_info_post), len(valid_info_post)).to(cfg.device)
#             for valid_info_pre, _, valid_hydra_pre, _ in tq:
#
#                 rl_in = [valid_info_pre, valid_info_post]
#                 chose_in = [valid_hydra_pre, valid_hydra_post]
#                 action, logp, value = rl_chose_hydra(rl, rl_in, chose_in, ....)
#                 hydra_pre, hydra_post = action
#
#                 bg_pairs = get_hydra_graph_label(valid_info_pre, valid_info_post, hydra_pre, hydra_post,
#                                                      pair=cfg.pair_match, valid=True)
#                 ret = match(bg_pairs, pairs=cfg.pair_match)
#
#                 B = valid_info_pre.shape[0]
#                 ret = ret.view(B, len(valid_info_post))
#                 all[num:num + B] = ret
#                 num += B
#
#         acc_list = []
#         for i in range(len(valid_info_post)):
#             row = all[i]
#             # 计算 row[i] 是不是最大的
#             if row[i] == torch.max(row):
#                 acc_list.append(1)
#             else:
#                 acc_list.append(0)
#
#
#         e_acc = np.mean(acc_list)
#         # e_acc_base = np.mean(acc_base_list) if len(acc_base_list) > 0 else float('nan')
#         print(f"[Valid] RL Acc: {e_acc:.3f} | ")
#
#         if e_acc >= bst_acc:
#             bst_acc = e_acc
#             early_stop = 0
#             torch.save(match.state_dict(), f'model_match_{e_acc}_{cfg.model_name}.pth')
#             bst_pth = f'model_match_{e_acc}_{cfg.model_name}.pth'
#             print(f'Best acc:{e_acc:.4f}  save model_para_{e_acc}_{cfg.model_name}.pth')
#         else:
#             early_stop += 1
#             if early_stop >= cfg.early_stop:
#                 break
#             # print(f'worse acc:{acc:.4f}')
#         # print(f'HR5:{bst5:.4f},HR10:{bst10:.4f}, HR50:{bst50:.4f}, NDCG:{bndcg:.7f}, HR10in50:{bst10in50:.4f}')
#
#     # model.load_state_dict(torch.load(bst_pth, map_location=cfg.device))
#
#     return
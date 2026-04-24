def reshape_pairs(pairs):
    if len(pairs) == 2:
        new_pairs = []
        for i in range(len(pairs[0])):
            new_pairs.append([pairs[0][i], pairs[1][i]])
        return new_pairs

    elif len(pairs[0])==2:
        pres, posts = [], []
        for pair in pairs:
            pre, post = pair
            pres.append(pre)
            posts.append(post)
        return [pres, posts]

def cmb_pre_post_traj(pairs):
    if not len(pairs) == 2:
        pairs = reshape_pairs(pairs)
    new = []
    new.extend(pairs[0])  # pre trajs
    new.extend(pairs[1])  # post trajs
    return new
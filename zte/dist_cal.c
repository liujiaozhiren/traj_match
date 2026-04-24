// traj_fusion.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <stdbool.h>

#define EARTH_RADIUS_M 6371008.8

// ----------------------------- 基础数据结构 -----------------------------
typedef struct {
    double t;     // 时间戳
    double lon;   // 经度 (deg)
    double lat;   // 纬度 (deg)
    double h;     // 高度（height=false 时可忽略）
    double conf;  // 置信度 ps（dB）
} Point;

typedef struct {
    Point* pts;
    int len;
} Traj;

typedef struct {
    Traj* arr;
    int len;
} TrajList;

typedef struct {
    double* data; // 行主序, 形状 m x n
    int m, n;
} DistMat;

// 工具：访问矩阵元素 D[i,j]
static inline double DMAT(const DistMat* D, int i, int j) { return D->data[(size_t)i * D->n + j]; }
static inline void   DMAT_SET(DistMat* D, int i, int j, double v) { D->data[(size_t)i * D->n + j] = v; }

// 内存释放
static void free_traj(Traj* t) {
    if (!t) return;
    free(t->pts);
    t->pts = NULL;
    t->len = 0;
}
static void free_trajlist(TrajList* L) {
    if (!L) return;
    for (int i = 0; i < L->len; ++i) free_traj(&L->arr[i]);
    free(L->arr);
    L->arr = NULL;
    L->len = 0;
}
static void free_distmat(DistMat* D) {
    if (!D) return;
    free(D->data);
    D->data = NULL; D->m = D->n = 0;
}

// ----------------------------- 数学/地理工具 -----------------------------
static inline double deg2rad(double d) { return d * (M_PI / 180.0); }

static double _haversine_m(double lat1, double lon1, double lat2, double lon2) {
    // 球面两点大圆距离（米），输入为度
    double phi1 = deg2rad(lat1), phi2 = deg2rad(lat2);
    double dphi = phi2 - phi1;
    double dlmb = deg2rad(lon2 - lon1);
    double a = sin(dphi/2.0)*sin(dphi/2.0) + cos(phi1)*cos(phi2)*sin(dlmb/2.0)*sin(dlmb/2.0);
    double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
    return EARTH_RADIUS_M * c;
}

static inline double db_to_linear(double db) {
    if (!isfinite(db)) return 0.0;
    return pow(10.0, db / 10.0); // 假设 dB 为功率刻度
}

static inline double round_to_gap_half_up(double t, int gap) {
    double q = t / (double)gap;
    double r = (q >= 0.0) ? floor(q + 0.5) : ceil(q - 0.5);
    return (double)((long long)r * (long long)gap);
}

// ----------------------------- 排序辅助 -----------------------------
typedef struct { double key_t; int idx; Point p; } AugPoint;

static int cmp_augpoint_by_t_idx(const void* a, const void* b) {
    const AugPoint* x = (const AugPoint*)a;
    const AugPoint* y = (const AugPoint*)b;
    if (x->key_t < y->key_t) return -1;
    if (x->key_t > y->key_t) return  1;
    // 次关键字：原始索引，保证可复现
    if (x->idx < y->idx) return -1;
    if (x->idx > y->idx) return  1;
    return 0;
}

static int cmp_point_by_t(const void* a, const void* b) {
    const Point* x = (const Point*)a;
    const Point* y = (const Point*)b;
    if (x->t < y->t) return -1;
    if (x->t > y->t) return  1;
    return 0;
}

// ----------------------------- 核心函数（与 Python 同名） -----------------------------
Traj traj_combine(const Traj* A, const Traj* B, int height) {
    // 两条轨迹按时间戳合并：相同时间戳 -> 按 dB 转线性做加权平均；不同时间戳 -> 原样保留
    // 约定：输入各自内部无重复时间戳（用户已说明）
    Traj out; out.len = 0; out.pts = NULL;
    int cap = (A->len + B->len > 0) ? (A->len + B->len) : 1;
    out.pts = (Point*)malloc(sizeof(Point) * (size_t)cap);

    // 为保证按时间戳有序，先复制并排序（若已保证有序可略去排序）
    Traj As, Bs; As.len = A->len; Bs.len = B->len;
    As.pts = (Point*)malloc(sizeof(Point) * (size_t)A->len);
    Bs.pts = (Point*)malloc(sizeof(Point) * (size_t)B->len);
    memcpy(As.pts, A->pts, sizeof(Point) * (size_t)A->len);
    memcpy(Bs.pts, B->pts, sizeof(Point) * (size_t)B->len);
    qsort(As.pts, (size_t)As.len, sizeof(Point), cmp_point_by_t);
    qsort(Bs.pts, (size_t)Bs.len, sizeof(Point), cmp_point_by_t);

    int i = 0, j = 0;
    while (i < As.len || j < Bs.len) {
        if (i < As.len && (j >= Bs.len || As.pts[i].t < Bs.pts[j].t)) {
            out.pts[out.len++] = As.pts[i++];
        } else if (j < Bs.len && (i >= As.len || Bs.pts[j].t < As.pts[i].t)) {
            out.pts[out.len++] = Bs.pts[j++];
        } else {
            // 相同时间戳
            Point p1 = As.pts[i++], p2 = Bs.pts[j++];
            double w1 = db_to_linear(p1.conf);
            double w2 = db_to_linear(p2.conf);
            double wsum = w1 + w2;

            Point m;
            m.t = p1.t; // == p2.t
            if (wsum > 0.0) {
                m.lon  = (p1.lon * w1 + p2.lon * w2) / wsum;
                m.lat  = (p1.lat * w1 + p2.lat * w2) / wsum;
                m.h    = height ? ((p1.h   * w1 + p2.h   * w2) / wsum) : 0.0;
                m.conf = 10.0 * log10(wsum);
            } else {
                // 权重全 0：退化为简单平均
                m.lon  = 0.5 * (p1.lon + p2.lon);
                m.lat  = 0.5 * (p1.lat + p2.lat);
                m.h    = height ? (0.5 * (p1.h + p2.h)) : 0.0;
                m.conf = -INFINITY;
            }
            out.pts[out.len++] = m;
        }
    }

    free(As.pts); free(Bs.pts);
    return out;
}

TrajList pre_proc(const TrajList* trajlist, int height, int timestampgap) {
    (void)height; // 字段结构已统一，height 仅影响语义；函数内不需特判
    if (timestampgap <= 0) timestampgap = 1;

    TrajList out; out.len = trajlist->len;
    out.arr = (Traj*)calloc((size_t)out.len, sizeof(Traj));

    for (int ti = 0; ti < trajlist->len; ++ti) {
        const Traj* T = &trajlist->arr[ti];
        if (T->len == 0) {
            out.arr[ti].pts = NULL; out.arr[ti].len = 0;
            continue;
        }
        AugPoint* buf = (AugPoint*)malloc(sizeof(AugPoint) * (size_t)T->len);
        for (int k = 0; k < T->len; ++k) {
            buf[k].idx   = k;          // 保持“后者覆盖前者”的依据
            buf[k].key_t = round_to_gap_half_up(T->pts[k].t, timestampgap);
            buf[k].p     = T->pts[k];
            buf[k].p.t   = buf[k].key_t; // 直接替换为取整后的时间戳
        }
        // 按 (key_t, idx) 排序
        qsort(buf, (size_t)T->len, sizeof(AugPoint), cmp_augpoint_by_t_idx);

        // 去重：相同 key_t 仅保留“最后一个”（即原轨迹里靠后的点）
        Point* merged = (Point*)malloc(sizeof(Point) * (size_t)T->len);
        int mlen = 0;
        int i = 0;
        while (i < T->len) {
            int j = i + 1;
            while (j < T->len && buf[j].key_t == buf[i].key_t) j++;
            // 保留该组里的最后一个（后者覆盖前者）
            merged[mlen++] = buf[j - 1].p;
            i = j;
        }
        free(buf);

        // 写入输出并按时间戳升序（已排序）
        out.arr[ti].pts = (Point*)realloc(merged, sizeof(Point) * (size_t)mlen);
        out.arr[ti].len = mlen;
    }
    return out;
}

double traj_dist(const Traj* t1, const Traj* t2, int height) {
    // 共同时间戳逐点距离平均；无共同时间戳返回 +inf
    if (t1->len == 0 || t2->len == 0) return INFINITY;

    // 确保按时间戳升序（复制 + 排序；如已有序可优化掉）
    Traj A = *t1, B = *t2;
    Point* a = (Point*)malloc(sizeof(Point) * (size_t)A.len);
    Point* b = (Point*)malloc(sizeof(Point) * (size_t)B.len);
    memcpy(a, A.pts, sizeof(Point) * (size_t)A.len);
    memcpy(b, B.pts, sizeof(Point) * (size_t)B.len);
    qsort(a, (size_t)A.len, sizeof(Point), cmp_point_by_t);
    qsort(b, (size_t)B.len, sizeof(Point), cmp_point_by_t);

    int i = 0, j = 0, cnt = 0;
    double total = 0.0;
    while (i < A.len && j < B.len) {
        double ta = a[i].t, tb = b[j].t;
        if (ta == tb) {
            double d_surface = _haversine_m(a[i].lat, a[i].lon, b[j].lat, b[j].lon);
            double d = height ? hypot(d_surface, a[i].h - b[j].h) : d_surface;
            total += d; cnt++;
            i++; j++;
        } else if (ta < tb) {
            i++;
        } else {
            j++;
        }
    }
    free(a); free(b);
    return (cnt == 0) ? INFINITY : (total / (double)cnt);
}

DistMat pairwise_traj_distance_np(const TrajList* A, const TrajList* B, int height) {
    DistMat D; D.m = A->len; D.n = B->len;
    size_t sz = (size_t)D.m * (size_t)D.n;
    D.data = (double*)malloc(sizeof(double) * (sz > 0 ? sz : 1));
    for (int i = 0; i < D.m; ++i) {
        for (int j = 0; j < D.n; ++j) {
            double d = traj_dist(&A->arr[i], &B->arr[j], height);
            DMAT_SET(&D, i, j, d);
        }
    }
    return D;
}

typedef struct { int i, j; double c; } Pair;
static int cmp_pair_cost(const void* a, const void* b) {
    const Pair* x = (const Pair*)a;
    const Pair* y = (const Pair*)b;
    if (x->c < y->c) return -1;
    if (x->c > y->c) return  1;
    if (x->i < y->i) return -1;
    if (x->i > y->i) return  1;
    if (x->j < y->j) return -1;
    if (x->j > y->j) return  1;
    return 0;
}

// 贪心匹配：返回 match_for_A（长 m），未匹配为 -1；total_cost 写入 *out_cost
int* greedy_assignment(const DistMat* D, double* out_cost) {
    int m = D->m, n = D->n;
    int mn = m * n;
    Pair* ps = (Pair*)malloc(sizeof(Pair) * (size_t)mn);
    int k = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double c = DMAT(D, i, j);
            if (!isfinite(c)) continue; // 跳过 inf/NaN
            ps[k].i = i; ps[k].j = j; ps[k].c = c; k++;
        }
    }
    qsort(ps, (size_t)k, sizeof(Pair), cmp_pair_cost);

    bool* used_row = (bool*)calloc((size_t)m, sizeof(bool));
    bool* used_col = (bool*)calloc((size_t)n, sizeof(bool));
    int* match = (int*)malloc(sizeof(int) * (size_t)m);
    for (int i = 0; i < m; ++i) match[i] = -1;

    double total_cost = 0.0;
    int matched = 0, target = (m < n ? m : n);
    for (int idx = 0; idx < k; ++idx) {
        int i = ps[idx].i, j = ps[idx].j;
        if (!used_row[i] && !used_col[j]) {
            used_row[i] = true; used_col[j] = true;
            match[i] = j;
            total_cost += ps[idx].c;
            matched++;
            if (matched == target) break;
        }
    }

    free(ps); free(used_row); free(used_col);
    if (out_cost) *out_cost = total_cost;
    return match;
}

TrajList traj_fusion(const TrajList* trajlistA,
                     const TrajList* trajlistB,
                     int height,
                     const char* assign,
                     double threshold) {
    // 预处理
    TrajList A_pp = pre_proc(trajlistA, height, 1000);
    TrajList B_pp = pre_proc(trajlistB, height, 1000);

    // 距离矩阵
    DistMat D = pairwise_traj_distance_np(&A_pp, &B_pp, height);

    // 匹配（仅支持贪心）
    if (assign && strcmp(assign, "greedy") != 0) {
        fprintf(stderr, "匈牙利算法需要外部库，当前仅支持 assign='greedy'，收到: %s\n",
                assign);
    }
    double total_cost = 0.0;
    int* match = greedy_assignment(&D, &total_cost);

    // 融合
    TrajList fused; fused.len = A_pp.len;
    fused.arr = (Traj*)calloc((size_t)fused.len, sizeof(Traj));
    for (int i = 0; i < fused.len; ++i) {
        int j = match[i];
        if (j == -1) { fused.arr[i].pts = NULL; fused.arr[i].len = 0; continue; }
        double dij = DMAT(&D, i, j);
        if (!isfinite(dij) || dij > threshold) { fused.arr[i].pts = NULL; fused.arr[i].len = 0; continue; }
        fused.arr[i] = traj_combine(&A_pp.arr[i], &B_pp.arr[j], height);
    }

    // 清理中间产物
    free(match);
    free_distmat(&D);
    free_trajlist(&A_pp);
    free_trajlist(&B_pp);

    return fused;
}

// 仅测试用的小工具：按时间戳构建“索引”（线性查找）
static int find_by_t(const Traj* T, double t) {
    for (int i = 0; i < T->len; ++i) if (T->pts[i].t == t) return i;
    return -1;
}

// ----------------------------- 自测 main（与 Python 用例等价） -----------------------------
int main(void) {
    // 构造数据
    TrajList A; A.len = 2; A.arr = (Traj*)calloc(2, sizeof(Traj));
    TrajList B; B.len = 1; B.arr = (Traj*)calloc(1, sizeof(Traj));

    // A1
    A.arr[0].len = 3;
    A.arr[0].pts = (Point*)malloc(sizeof(Point) * 3);
    A.arr[0].pts[0] = (Point){1000, 0.0, 0.0, 10.0, 0.0};
    A.arr[0].pts[1] = (Point){2000, 0.0, 0.0, 12.0, 0.0};
    A.arr[0].pts[2] = (Point){2999, 0.0, 0.0, 14.0, 0.0};

    // B1
    B.arr[0].len = 3;
    B.arr[0].pts = (Point*)malloc(sizeof(Point) * 3);
    B.arr[0].pts[0] = (Point){1000, 0.0, 0.0, 11.0, 0.0};
    B.arr[0].pts[1] = (Point){2000, 0.0, 0.0, 22.0, 10.0};
    B.arr[0].pts[2] = (Point){3000, 0.0, 0.0, 16.0, 0.0};

    // A2
    A.arr[1].len = 3;
    A.arr[1].pts = (Point*)malloc(sizeof(Point) * 3);
    A.arr[1].pts[0] = (Point){1901, 1.0, 0.0, 0.0, 0.0};
    A.arr[1].pts[1] = (Point){2099, 2.0, 0.0, 0.0, 10.0}; // -> 2000 后者覆盖前者
    A.arr[1].pts[2] = (Point){3000, 5.0, 0.0, 0.0, 0.0};

    // pre_proc
    TrajList A_pp = pre_proc(&A, /*height=*/1, /*timestampgap=*/1000);
    TrajList B_pp = pre_proc(&B, /*height=*/1, /*timestampgap=*/1000);

    // A1: 2999 -> 3000
    assert(A_pp.arr[0].len == 3);
    assert(A_pp.arr[0].pts[0].t == 1000 && A_pp.arr[0].pts[1].t == 2000 && A_pp.arr[0].pts[2].t == 3000);

    // A2: 1901/2099 -> 都为 2000，后者覆盖 => lon=2.0
    assert(A_pp.arr[1].len == 2);
    assert(A_pp.arr[1].pts[0].t == 2000 && A_pp.arr[1].pts[1].t == 3000);
    {
        int idx2000 = find_by_t(&A_pp.arr[1], 2000);
        assert(idx2000 >= 0);
        assert(fabs(A_pp.arr[1].pts[idx2000].lon - 2.0) < 1e-12);
    }

    // 距离 + 贪心
    DistMat D = pairwise_traj_distance_np(&A_pp, &B_pp, /*height=*/1);
    assert(D.m == 2 && D.n == 1);
    assert(DMAT(&D, 0, 0) < DMAT(&D, 1, 0));

    double cost = 0.0;
    int* match = greedy_assignment(&D, &cost);
    assert(match[0] == 0 && match[1] == -1);

    // 融合
    TrajList fused = traj_fusion(&A, &B, /*height=*/1, "greedy", 1e5);
    assert(fused.len == 2);
    assert(fused.arr[0].len > 0 && fused.arr[1].len == 0);

    // 校验合并数值
    const Traj* f0 = &fused.arr[0];
    int i1000 = find_by_t(f0, 1000);
    int i2000 = find_by_t(f0, 2000);
    int i3000 = find_by_t(f0, 3000);
    assert(i1000 >= 0 && i2000 >= 0 && i3000 >= 0);

    // t=1000: h=(10+11)/2=10.5; conf=10*log10(1+1)
    assert(fabs(f0->pts[i1000].h - 10.5) < 1e-6);
    assert(fabs(f0->pts[i1000].conf - 10.0*log10(2.0)) < 1e-6);

    // t=2000: 权重 1:10 -> h=(12*1 + 22*10)/11 = 232/11; conf=10*log10(11)
    assert(fabs(f0->pts[i2000].h - (232.0/11.0)) < 1e-6);
    assert(fabs(f0->pts[i2000].conf - 10.0*log10(11.0)) < 1e-6);

    // t=3000: h=(14+16)/2=15; conf=10*log10(2)
    assert(fabs(f0->pts[i3000].h - 15.0) < 1e-6);
    assert(fabs(f0->pts[i3000].conf - 10.0*log10(2.0)) < 1e-6);

    // 进一步 sanity check
    assert(DMAT(&D, 0, 0) < 1e5);

    printf("All tests passed ✅\n");
    printf("D(0,0) = %.6f\n", DMAT(&D, 0, 0));
    printf("match_for_A = [%d, %d], total_cost = %.6f\n", match[0], match[1], cost);
    printf("fused[0] length = %d\n", fused.arr[0].len);

    // 清理
    free(match);
    free_distmat(&D);
    free_trajlist(&A_pp);
    free_trajlist(&B_pp);
    free_trajlist(&fused);
    free_trajlist(&A);
    free_trajlist(&B);

    return 0;
}

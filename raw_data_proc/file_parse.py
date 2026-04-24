import os.path
import pickle
from collections import defaultdict

from numpy import float64
from scapy.all import rdpcap, IP, UDP
from scapy.layers.inet import defragment
from typing import Dict, List, Tuple, Any, Optional
import struct

from ue_parse_util import parse_ns_payload

ALLOWED_GTPU_DATA_TYPES = {0xFF, 0x28}
GTPU_PORT = 2152          # 标准 GTP‑U 端口
def _strip_gtpu_header(data: bytes) -> Tuple[Optional[bytes], Optional[int]]:
    """
    从 UDP 负载中剥离 GTP‑U 头，返回 (inner_payload, teid)。
    若报文格式异常或 MessageType 不在允许范围，则返回 (None, None)。
    兼容含/不含可选 4 字节及扩展头的情况；支持 Python 3.8+。
    """
    if len(data) < 8:
        return None, None

    flags, msg_type, length = struct.unpack_from(">BBH", data, 0)
    teid, = struct.unpack_from(">I", data, 4)

    # 仅处理承载用户数据的消息类型
    if msg_type not in ALLOWED_GTPU_DATA_TYPES:
        return None, None

    total_msg_len = 8 + length            # 固定头 8 B + Length
    if len(data) < total_msg_len:         # 报文被截断
        return None, None

    # ------------------------------------------------------------
    # 找到真正业务负载起点：8 B 固定头 + 可选 4 B + 扩展头链
    # ------------------------------------------------------------
    pos = 8                               # 跳过固定头
    has_seq  = flags & 0x02               # S 位
    has_npdu = flags & 0x01               # PN 位
    has_ext  = flags & 0x04               # E 位

    if has_seq or has_npdu or has_ext:    # 可选 4 字节齐头并进
        pos += 4

    if has_ext:
        # 可能存在一串扩展头；每块 Len 单位=4B，不含前两字节
        while True:
            if pos + 2 > total_msg_len:
                return None, None         # 格式错误
            ext_len_units = data[pos]     # 4‑byte units
            next_type     = data[pos + 1 + ext_len_units * 4]
            pos += 2 + ext_len_units * 4
            if next_type == 0:            # 0 代表扩展头链结束
                break

    if pos > total_msg_len:               # 防护
        return None, None

    inner_payload = data[pos: total_msg_len]
    return inner_payload, teid

def extract_udp_payloads(pcap_path: str) -> Dict[Tuple[str, str], List[Any]]:
    """
    1. 重组 IP 分片
    2. 剥 UDP → GTP‑U 头
    3. 调用 parse_sf_af_payload 解析业务 TLV
    4. 按 (src_ip, dst_ip) 聚合结果
    """
    packets = rdpcap(pcap_path)
    reassembled = list(defragment(packets))

    buckets: Dict[Tuple[str, str], List[Any]] = defaultdict(list)

    for pkt in reassembled:
        if not (pkt.haslayer(IP) and pkt.haslayer(UDP)):
            continue

        udp = pkt[UDP]
        # ✨ 只保留 src/dst 任一端口为 2152 的 GTP‑U 报文
        if udp.sport != GTPU_PORT and udp.dport != GTPU_PORT:
            continue

        udp_data  = bytes(udp.payload)
        if not udp_data:
            continue
        inner, teid = _strip_gtpu_header(udp_data)
        if inner is None:                      # 非 G‑PDU 或格式异常
            continue

        #parsed = inner
        parsed = parse_ns_payload(inner)
        # if parsed is None:                     # 未知 MessageType
        #     continue

        key = (pkt[IP].src, pkt[IP].dst)   # 如需按 TEID 细分，可改为 (src,dst,teid)
        buckets[key].append(parsed)

    return buckets

def parse_pcap(path):
    #path = 'sz-sf4.pcapng'
    path = str(path)
    #path.tostr()
    if not os.path.exists(path+'tmp.pkl'):
        ret = extract_udp_payloads(path)
        pickle.dump(ret, open(path+'tmp.pkl','wb'))
    else:
        ret = pickle.load(open(path+'tmp.pkl','rb'))
    buckets={}
    cnt=0
    for _, ret0 in ret.items():
        for item in ret0:
            if item['msgType'] != 0:
                continue
            for group in item['detectedObjectGroupList']:  # List<ObjectGroup>
                g_id = group.get('gNBID', -1)
                for obj in group.get("detectedObjectList", []):
                    oid = obj["objectID"]
                    tp = obj["objectTrackPoint"]
                    loc = tp["location"]
                    spd = tp.get("speedXYZ", {})  # 可能不存在
                    ot = obj['objectType']
                    cfd = obj['objectConfidence']
                    # assert ot == 1
                    if oid not in buckets:
                        buckets[oid]={'type': ot}
                    if g_id not in buckets[oid]:
                        buckets[oid][g_id] = []
                    buckets[oid][g_id].append([tp["timestamp"],
                                                  float64(loc["longitude"]),
                                                  float64(loc["latitude"]),
                                                  float64(loc["altitude"]),
                                                  float64(tp['speed']),
                                                  float64(tp.get("speedX", float64("nan"))),
                                                  float64(tp.get("speedY", float64("nan"))),
                                                  float64(tp.get("speedZ", float64("nan"))),
                                                  g_id,])
    result = []
    single_traj = []
    multi_traj = {}
    for oid, track in buckets.items():
        cmb_traj = []
        for g_id, traj in track.items():
            if g_id == 'type':
                continue
            traj.sort(key=lambda x: x[0])
            cmb_traj.extend(traj)
            if len(track) == 2:
                single_traj.append(traj)
            if len(track) >= 3:
                if oid not in multi_traj:
                    multi_traj[oid] = []
                multi_traj[oid].append(traj)
        cmb_traj.sort(key=lambda x: x[0])
        result.append(cmb_traj)
    return result, single_traj, multi_traj
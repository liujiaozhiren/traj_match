# -*- coding: utf-8 -*-
"""
Ns‑side 内部负载（TLV：Tag‑2B | Len‑2B | Value）解析器
兼容两种消息：
    0 → NsSenseMeasurementReport
    1 → NsSenseHistoryUpdate
Python 3.8+；纯标准库。
"""
import struct, uuid
from decimal import Decimal, getcontext
from typing import Dict, List, Any, Tuple, Optional

getcontext().prec = 17          # float64 → Decimal 足够精度

# ──────────────────────────────────────────────────────────────────
# TLV Tag 常量
# ──────────────────────────────────────────────────────────────────
T = {
    "msgType": 0x0001, "msgCnt": 0x0002, "msgTime": 0x0003,
    "ObjectGroup": 0x0004, "gNBID": 0x0005, "trpID": 0x0006,
    "detectorPos": 0x0007, "Object": 0x0008, "objectID": 0x0009,
    "objectType": 0x000A, "TrackPoint": 0x000B, "timestamp": 0x000C,
    "location": 0x000D, "speed": 0x000E, "speedX": 0x000F,
    "speedY": 0x0010, "speedZ": 0x0011, "plmn": 0x0015,
    "gNBIDLen": 0x0016, "objectConfidence": 0x0017, "cellId": 0x0018,
    "beamId": 0x0019, "beamId_H": 0x001A, "beamId_V": 0x001B,
    "ts": 0x001C, "f": 0x001D, "phi": 0x001E, "theta": 0x001F,
    "sinr": 0x0020, "ps": 0x0021, "ni": 0x0022, "distance": 0x0023,
    "radial_speed": 0x0024, "point_type": 0x0025,
}

_UINT32 = {T["msgType"], T["msgCnt"], T["gNBID"], T["trpID"],
           T["cellId"], T["beamId"], T["beamId_H"], T["beamId_V"]}
_UINT64 = {T["msgTime"], T["timestamp"]}
_FLOAT64 = {T["speed"], T["speedX"], T["speedY"], T["speedZ"],
            T["objectConfidence"], T["ts"], T["f"], T["phi"],
            T["theta"], T["sinr"], T["ps"], T["ni"],
            T["distance"], T["radial_speed"]}
_UINT8 = {T["gNBIDLen"], T["objectType"], T["point_type"]}

# ──────────────────────────────────────────────────────────────────
# 顶层解析
# ──────────────────────────────────────────────────────────────────
def parse_ns_payload(buf: bytes) -> Dict[str, Any]:
    tlvs, _ = _parse_tlvs(buf)
    msg: Dict[str, Any] = {}
    for tlv in tlvs:
        tag = tlv["tag"]
        if tag == T["msgType"]:
            msg["msgType"] = tlv["value"]
        elif tag == T["msgCnt"]:
            msg["msgCnt"] = tlv["value"]
        elif tag == T["msgTime"]:
            msg["msgTime"] = tlv["value"]
        elif tag == T["ObjectGroup"]:
            msg.setdefault("detectedObjectGroupList", []).append(
                _parse_object_group(tlv["raw"]))
    return msg

# ──────────────────────────────────────────────────────────────────
# TLV helpers
# ──────────────────────────────────────────────────────────────────
def _parse_tlvs(buf: bytes, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
    pos, end = 0, limit or len(buf)
    out: List[Dict[str, Any]] = []
    while pos + 4 <= end:
        tag, length = struct.unpack_from(">HH", buf, pos)
        pos += 4
        value_raw = buf[pos: pos + length]
        pos += length
        out.append({"tag": tag,
                    "value": _decode_primitive(tag, value_raw),
                    "raw": value_raw})
    return out, pos

def _decode_primitive(tag: int, raw: bytes) -> Any:
    if tag in _UINT32 and len(raw) == 4:
        return struct.unpack(">I", raw)[0]
    if tag in _UINT64 and len(raw) == 8:
        return struct.unpack(">Q", raw)[0]
    if tag in _UINT8 and len(raw) == 1:
        return raw[0]
    if tag in _FLOAT64 and len(raw) == 8:
        return Decimal(struct.unpack(">d", raw)[0])
    if tag == T["objectID"] and len(raw) == 16:
        return str(uuid.UUID(bytes=raw))
    # 默认按 utf‑8 字符串处理
    try:
        return raw.decode()
    except UnicodeDecodeError:
        return raw

# ──────────────────────────────────────────────────────────────────
# 复合结构解析
# ──────────────────────────────────────────────────────────────────
def _parse_position(raw: bytes) -> Dict[str, Decimal]:
    lon, lat, alt = struct.unpack(">ddd", raw)
    return {"longitude": Decimal(lon),
            "latitude":  Decimal(lat),
            "altitude":  Decimal(alt)}

def _parse_track_point(raw: bytes) -> Dict[str, Any]:
    tp: Dict[str, Any] = {}
    for tlv in _parse_tlvs(raw)[0]:
        tag = tlv["tag"]
        if tag == T["timestamp"]:
            tp["timestamp"] = tlv["value"]
        elif tag == T["location"]:
            tp["location"] = _parse_position(tlv["raw"])
        elif tag in {T["speed"], T["speedX"], T["speedY"], T["speedZ"]}:
            name = {T["speed"]: "speed", T["speedX"]: "speedX",
                    T["speedY"]: "speedY", T["speedZ"]: "speedZ"}[tag]
            tp[name] = tlv["value"]
        else:
            tp.setdefault("extra", {})[tag] = tlv["value"]
    return tp

def _parse_object(raw: bytes) -> Dict[str, Any]:
    obj: Dict[str, Any] = {}
    for tlv in _parse_tlvs(raw)[0]:
        tag = tlv["tag"]
        if tag == T["objectID"]:
            obj["objectID"] = tlv["value"]
        elif tag == T["objectType"]:
            obj["objectType"] = tlv["value"]
        elif tag == T["objectConfidence"]:
            obj["objectConfidence"] = tlv["value"]
        elif tag == T["TrackPoint"]:
            obj["objectTrackPoint"] = _parse_track_point(tlv["raw"])
        else:
            obj.setdefault("extra", {})[tag] = tlv["value"]
    return obj

def _parse_object_group(raw: bytes) -> Dict[str, Any]:
    grp: Dict[str, Any] = {}
    for tlv in _parse_tlvs(raw)[0]:
        tag = tlv["tag"]
        if tag == T["plmn"]:
            grp["plmn"] = tlv["value"]
        elif tag == T["gNBIDLen"]:
            grp["gNBIDLen"] = tlv["value"]
        elif tag == T["gNBID"]:
            grp["gNBID"] = tlv["value"]
        elif tag == T["trpID"]:
            grp["trpID"] = tlv["value"]
        elif tag == T["cellId"]:
            grp["cellId"] = tlv["value"]
        elif tag == T["detectorPos"]:
            grp["detectorPos"] = _parse_position(tlv["raw"])
        elif tag == T["Object"]:
            grp.setdefault("detectedObjectList", []).append(
                _parse_object(tlv["raw"]))
        else:
            grp.setdefault("extra", {})[tag] = tlv["value"]
    return grp
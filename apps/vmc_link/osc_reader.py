"""最小 OSC 解码器：只覆盖 VMC Protocol 用到的类型，供监听工具与自检共用。

按 https://protocol.vmc.info/english 的编码规则实现（UTF-8 + NUL + 4 字节对齐，
'i'/'f' 大端，'s' 字符串），并支持 bundle 解包（接收端本来就必须能处理）。
"""
import struct


def read_string(data, offset):
    end = data.index(b"\x00", offset)
    text = data[offset:end].decode("utf-8", errors="replace")
    offset = end + 1
    offset += (4 - offset % 4) % 4
    return text, offset


def parse_message(data):
    """解析单条 OSC 消息 → (address, args)。"""
    address, offset = read_string(data, 0)
    typetags, offset = read_string(data, offset)
    args = []
    for tag in typetags.lstrip(","):
        if tag == "i":
            args.append(struct.unpack_from(">i", data, offset)[0])
            offset += 4
        elif tag == "f":
            args.append(struct.unpack_from(">f", data, offset)[0])
            offset += 4
        elif tag == "d":
            args.append(struct.unpack_from(">d", data, offset)[0])
            offset += 8
        elif tag == "s":
            value, offset = read_string(data, offset)
            args.append(value)
        elif tag in ("T", "F"):
            args.append(tag == "T")
        else:
            raise ValueError(f"不支持的 OSC 类型标记: {tag!r}")
    return address, args


def parse_packet(data):
    """解析一个 UDP 包 → [(address, args), ...]；bundle 会被展开。"""
    if not data.startswith(b"#bundle\x00"):
        return [parse_message(data)]

    messages = []
    offset = 16  # "#bundle\0"(8) + timetag(8)
    while offset + 4 <= len(data):
        size = struct.unpack_from(">i", data, offset)[0]
        offset += 4
        chunk = data[offset:offset + size]
        offset += size
        messages.extend(parse_packet(chunk))
    return messages

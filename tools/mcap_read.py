"""Minimal MCAP reader (no dependencies beyond optional zstandard/lz4)."""
import struct, sys

def records(buf):
    i = 8
    while i + 9 <= len(buf):
        op = buf[i]; ln = struct.unpack_from('<Q', buf, i + 1)[0]
        yield op, buf[i + 9:i + 9 + ln]
        i += 9 + ln

def read(path):
    data = open(path, 'rb').read()
    assert data[:8] == b'\x89MCAP0\r\n', 'not mcap'
    schemas, channels, msgs = {}, {}, []
    def s_(b, o):
        n = struct.unpack_from('<I', b, o)[0]; return b[o + 4:o + 4 + n].decode(), o + 4 + n
    def handle(op, b):
        if op == 0x03:
            sid = struct.unpack_from('<H', b, 0)[0]; name, o = s_(b, 2); enc, o = s_(b, o)
            n = struct.unpack_from('<I', b, o)[0]; schemas[sid] = (name, b[o + 4:o + 4 + n].decode(errors='replace'))
        elif op == 0x04:
            cid, sid = struct.unpack_from('<HH', b, 0); topic, o = s_(b, 4); channels[cid] = (topic, sid)
        elif op == 0x05:
            cid, seq, lt, pt = struct.unpack_from('<HIQQ', b, 0); msgs.append((cid, lt, b[22:]))
        elif op == 0x06:
            o = 8 + 8 + 8 + 4
            comp, o = s_(b, o); rlen = struct.unpack_from('<Q', b, o)[0]; raw = b[o + 8:o + 8 + rlen]
            if comp == 'zstd':
                import zstandard; raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 31)
            elif comp == 'lz4':
                import lz4.frame; raw = lz4.frame.decompress(raw)
            for op2, b2 in records(b'\x00' * 8 + raw):
                handle(op2, b2)
    for op, b in records(data):
        handle(op, b)
    return schemas, channels, msgs

if __name__ == '__main__':
    sc, ch, ms = read(sys.argv[1])
    from collections import Counter
    c = Counter(ch[m[0]][0] for m in ms)
    for k, v in c.items(): print(v, k)

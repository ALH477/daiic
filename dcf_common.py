import struct
import time
import socket
from dataclasses import dataclass
from typing import Optional, Tuple

# --- Protocol Constants ---
HEADER_FORMAT = ">B I Q I"  # Big-Endian: Type(u8), Seq(u32), Ts(u64), Len(u32)
HEADER_SIZE = 17

# Message Types
MSG_HEARTBEAT = 0x01
MSG_TASK      = 0x02
MSG_RESULT    = 0x03
MSG_ERROR     = 0xFF

@dataclass
class DCFMessage:
    msg_type: int
    sequence: int
    timestamp: int
    payload: bytes

    @staticmethod
    def current_timestamp_micros() -> int:
        return int(time.time() * 1_000_000)

    def serialize(self) -> bytes:
        payload_len = len(self.payload)
        header = struct.pack(
            HEADER_FORMAT,
            self.msg_type,
            self.sequence,
            self.timestamp,
            payload_len
        )
        return header + self.payload

    @classmethod
    def deserialize(cls, data: bytes) -> Optional['DCFMessage']:
        if len(data) < HEADER_SIZE:
            return None
        try:
            msg_type, sequence, timestamp, payload_len = struct.unpack(
                HEADER_FORMAT, data[:HEADER_SIZE]
            )
            if len(data) < HEADER_SIZE + payload_len:
                return None
            payload = data[HEADER_SIZE : HEADER_SIZE + payload_len]
            return cls(msg_type, sequence, timestamp, payload)
        except struct.error:
            return None

class AsyncUDPSocket:
    """Non-blocking UDP wrapper."""
    def __init__(self, port: int, bind_ip: str = "0.0.0.0"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_ip, port))
        self.sock.setblocking(False)
        print(f"[Net] Listening on {bind_ip}:{port}")

    def send(self, msg: DCFMessage, addr: Tuple[str, int]):
        self.sock.sendto(msg.serialize(), addr)

    def recv(self) -> Optional[Tuple[DCFMessage, Tuple[str, int]]]:
        try:
            data, addr = self.sock.recvfrom(65536)
            msg = DCFMessage.deserialize(data)
            if msg: return msg, addr
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"[Net] Recv Error: {e}")
        return None

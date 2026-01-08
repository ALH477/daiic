import time
import logging
import threading
from dcf_common import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s [HEAD] %(message)s')
logger = logging.getLogger("HeadNode")

PORT_CLIENT_INGRESS = 7777
PORT_WORKER_BUS     = 7778

class WorkerRegistry:
    def __init__(self):
        self.workers = {} 
        self.lock = threading.Lock()
        self.rr_index = 0

    def register(self, addr):
        with self.lock:
            if addr not in self.workers:
                logger.info(f"New Worker Registered: {addr}")
            self.workers[addr] = time.time()

    def get_next_worker(self):
        with self.lock:
            now = time.time()
            active = [addr for addr, ts in self.workers.items() if now - ts < 10]
            if not active:
                return None
            self.rr_index = (self.rr_index + 1) % len(active)
            return active[self.rr_index]

def main():
    sock_client = AsyncUDPSocket(PORT_CLIENT_INGRESS)
    sock_internal = AsyncUDPSocket(PORT_WORKER_BUS)
    registry = WorkerRegistry()
    request_map = {} 

    logger.info("HydraMesh Head Online. Waiting for traffic...")

    while True:
        # Handle Internal (Workers)
        packet = sock_internal.recv()
        if packet:
            msg, addr = packet
            if msg.msg_type == MSG_HEARTBEAT:
                try:
                    worker_port = int(msg.payload.decode())
                    worker_addr = (addr[0], worker_port)
                    registry.register(worker_addr)
                except: pass
            elif msg.msg_type == MSG_RESULT:
                client_addr = request_map.pop(msg.sequence, None)
                if client_addr:
                    resp = DCFMessage(MSG_TASK, msg.sequence, msg.timestamp, msg.payload)
                    sock_client.send(resp, client_addr)
                    logger.info(f"Task {msg.sequence} completed -> {client_addr}")

        # Handle External (Clients)
        packet = sock_client.recv()
        if packet:
            msg, client_addr = packet
            request_map[msg.sequence] = client_addr
            worker_addr = registry.get_next_worker()
            if worker_addr:
                fwd_msg = DCFMessage(MSG_TASK, msg.sequence, msg.timestamp, msg.payload)
                sock_internal.send(fwd_msg, worker_addr)
                logger.info(f"Routed Task {msg.sequence} -> {worker_addr}")
            else:
                logger.error("No workers available!")

        time.sleep(0.0001)

if __name__ == "__main__":
    main()

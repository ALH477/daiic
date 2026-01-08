#!/usr/bin/env python3
"""
HydraMesh Head Controller v5.1.0
Central router for the DCF AI Inference Cluster

Copyright (c) 2026 DeMoD LLC. All rights reserved.
Licensed under BSD-3-Clause.
"""

import time
import signal
import threading
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Set
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor

from dcf_common import (
    EventDrivenUDPSocket, DCFMessage, MessageType, ErrorCode,
    NodeMetrics, ChunkAssembler, parse_chunk, setup_logging,
    DEFAULT_WORKER_TIMEOUT, DEFAULT_REQUEST_TIMEOUT
)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

PORT_CLIENT_INGRESS = int(os.getenv("DCF_CLIENT_PORT", "7777"))
PORT_WORKER_BUS = int(os.getenv("DCF_WORKER_PORT", "7778"))
PORT_HEALTH_HTTP = int(os.getenv("DCF_HEALTH_PORT", "8080"))

WORKER_TIMEOUT = float(os.getenv("DCF_WORKER_TIMEOUT", str(DEFAULT_WORKER_TIMEOUT)))
REQUEST_TIMEOUT = float(os.getenv("DCF_REQUEST_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT)))
MAX_PENDING_REQUESTS = int(os.getenv("DCF_MAX_PENDING", "10000"))

logger = setup_logging("HEAD")


# ═══════════════════════════════════════════════════════════════════════════════
# Worker Registry with Health Tracking
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorkerInfo:
    """Tracks individual worker state."""
    addr: Tuple[str, int]
    last_heartbeat: float
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_latency_ms: float = 0.0
    current_task: Optional[int] = None  # Sequence number of active task
    
    def is_healthy(self, timeout: float = WORKER_TIMEOUT) -> bool:
        return time.time() - self.last_heartbeat < timeout
    
    def is_busy(self) -> bool:
        return self.current_task is not None


class WorkerRegistry:
    """
    Thread-safe worker registry with round-robin load balancing.
    
    Features:
    - Automatic stale worker pruning
    - Load-aware routing (prefers idle workers)
    - Health metrics per worker
    """
    
    def __init__(self):
        self._workers: Dict[Tuple[str, int], WorkerInfo] = {}
        self._lock = threading.RLock()
        self._rr_index = 0
        self._logger = setup_logging("REGISTRY")

    def register(self, addr: Tuple[str, int]) -> bool:
        """Register or refresh a worker. Returns True if new registration."""
        with self._lock:
            is_new = addr not in self._workers
            if is_new:
                self._workers[addr] = WorkerInfo(addr=addr, last_heartbeat=time.time())
                self._logger.info(f"✓ Worker registered: {addr[0]}:{addr[1]}")
            else:
                self._workers[addr].last_heartbeat = time.time()
            return is_new

    def get_worker(self, prefer_idle: bool = True) -> Optional[Tuple[str, int]]:
        """
        Get next available worker using smart load balancing.
        
        Priority: idle workers first, then round-robin among busy ones.
        """
        with self._lock:
            now = time.time()
            healthy = [w for w in self._workers.values() if w.is_healthy(WORKER_TIMEOUT)]
            
            if not healthy:
                return None
            
            # Prefer idle workers
            if prefer_idle:
                idle = [w for w in healthy if not w.is_busy()]
                if idle:
                    return idle[0].addr
            
            # Round-robin fallback
            self._rr_index = (self._rr_index + 1) % len(healthy)
            return healthy[self._rr_index].addr

    def assign_task(self, addr: Tuple[str, int], sequence: int):
        """Mark worker as busy with a task."""
        with self._lock:
            if addr in self._workers:
                self._workers[addr].current_task = sequence

    def complete_task(self, addr: Tuple[str, int], latency_ms: float, success: bool = True):
        """Mark task as complete on worker."""
        with self._lock:
            if addr in self._workers:
                worker = self._workers[addr]
                worker.current_task = None
                if success:
                    worker.tasks_completed += 1
                    # Rolling average
                    n = worker.tasks_completed
                    worker.avg_latency_ms = ((n - 1) * worker.avg_latency_ms + latency_ms) / n
                else:
                    worker.tasks_failed += 1

    def prune_stale(self) -> int:
        """Remove workers that haven't sent heartbeats. Returns count removed."""
        with self._lock:
            now = time.time()
            stale = [addr for addr, w in self._workers.items() 
                     if not w.is_healthy(WORKER_TIMEOUT)]
            for addr in stale:
                self._logger.warning(f"✗ Worker stale, removing: {addr[0]}:{addr[1]}")
                del self._workers[addr]
            return len(stale)

    def get_stats(self) -> dict:
        """Get registry statistics."""
        with self._lock:
            healthy = [w for w in self._workers.values() if w.is_healthy()]
            busy = [w for w in healthy if w.is_busy()]
            return {
                "total_workers": len(self._workers),
                "healthy_workers": len(healthy),
                "busy_workers": len(busy),
                "workers": [
                    {
                        "addr": f"{w.addr[0]}:{w.addr[1]}",
                        "healthy": w.is_healthy(),
                        "busy": w.is_busy(),
                        "tasks_completed": w.tasks_completed,
                        "avg_latency_ms": round(w.avg_latency_ms, 2)
                    }
                    for w in self._workers.values()
                ]
            }


# ═══════════════════════════════════════════════════════════════════════════════
# Request Tracking with TTL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PendingRequest:
    """Tracks an in-flight request."""
    client_addr: Tuple[str, int]
    worker_addr: Optional[Tuple[str, int]]
    created_at: float
    payload_size: int
    
    def is_expired(self, timeout: float = REQUEST_TIMEOUT) -> bool:
        return time.time() - self.created_at > timeout


class RequestTracker:
    """
    Manages pending requests with automatic timeout cleanup.
    
    Prevents memory leaks from orphaned requests.
    """
    
    def __init__(self, max_pending: int = MAX_PENDING_REQUESTS):
        self._requests: Dict[int, PendingRequest] = {}
        self._lock = threading.Lock()
        self._max_pending = max_pending
        self._logger = setup_logging("REQUESTS")

    def add(self, sequence: int, client_addr: Tuple[str, int], 
            worker_addr: Optional[Tuple[str, int]], payload_size: int) -> bool:
        """Add a pending request. Returns False if at capacity."""
        with self._lock:
            if len(self._requests) >= self._max_pending:
                self._logger.warning("Request queue at capacity!")
                return False
            self._requests[sequence] = PendingRequest(
                client_addr=client_addr,
                worker_addr=worker_addr,
                created_at=time.time(),
                payload_size=payload_size
            )
            return True

    def complete(self, sequence: int) -> Optional[PendingRequest]:
        """Remove and return a completed request."""
        with self._lock:
            return self._requests.pop(sequence, None)

    def cleanup_expired(self) -> int:
        """Remove expired requests. Returns count removed."""
        with self._lock:
            now = time.time()
            expired = [seq for seq, req in self._requests.items() if req.is_expired()]
            for seq in expired:
                self._logger.warning(f"Request {seq} expired after {REQUEST_TIMEOUT}s")
                del self._requests[seq]
            return len(expired)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "pending_requests": len(self._requests),
                "max_pending": self._max_pending
            }


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP Health Server
# ═══════════════════════════════════════════════════════════════════════════════

class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health checks and metrics."""
    
    registry: WorkerRegistry = None
    tracker: RequestTracker = None
    metrics: NodeMetrics = None
    
    def log_message(self, format, *args):
        pass  # Suppress default logging
    
    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "healthy", "timestamp": time.time()})
        elif self.path == "/ready":
            stats = self.registry.get_stats() if self.registry else {}
            ready = stats.get("healthy_workers", 0) > 0
            status = 200 if ready else 503
            self._send_json({"ready": ready, "workers": stats.get("healthy_workers", 0)}, status)
        elif self.path == "/metrics":
            data = {
                "node": self.metrics.to_dict() if self.metrics else {},
                "registry": self.registry.get_stats() if self.registry else {},
                "requests": self.tracker.get_stats() if self.tracker else {}
            }
            self._send_json(data)
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())


def start_health_server(registry: WorkerRegistry, tracker: RequestTracker, 
                        metrics: NodeMetrics, port: int = PORT_HEALTH_HTTP):
    """Start HTTP health server in background thread."""
    HealthHandler.registry = registry
    HealthHandler.tracker = tracker
    HealthHandler.metrics = metrics
    
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health server listening on http://0.0.0.0:{port}")
    return server


# ═══════════════════════════════════════════════════════════════════════════════
# Head Controller Main Loop
# ═══════════════════════════════════════════════════════════════════════════════

class HeadController:
    """
    Central routing controller for the HydraMesh cluster.
    
    Responsibilities:
    - Accept client inference requests
    - Load-balance tasks across workers
    - Track request lifecycle
    - Handle chunked responses
    - Provide health/metrics endpoints
    """
    
    def __init__(self):
        self.registry = WorkerRegistry()
        self.tracker = RequestTracker()
        self.metrics = NodeMetrics()
        
        self.sock_client = EventDrivenUDPSocket(PORT_CLIENT_INGRESS)
        self.sock_internal = EventDrivenUDPSocket(PORT_WORKER_BUS)
        
        self._shutdown = threading.Event()
        self._chunk_assemblers: Dict[int, ChunkAssembler] = {}
        
        # Start background maintenance
        self._maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        
    def start(self):
        """Start the head controller."""
        logger.info("═" * 60)
        logger.info("  HydraMesh Head Controller v5.1.0")
        logger.info("═" * 60)
        logger.info(f"  Client Port:   UDP {PORT_CLIENT_INGRESS}")
        logger.info(f"  Worker Port:   UDP {PORT_WORKER_BUS}")
        logger.info(f"  Health Port:   HTTP {PORT_HEALTH_HTTP}")
        logger.info("═" * 60)
        
        # Start health server
        self._health_server = start_health_server(
            self.registry, self.tracker, self.metrics
        )
        
        # Start maintenance thread
        self._maintenance_thread.start()
        
        # Main event loop
        self._run()

    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Initiating graceful shutdown...")
        self._shutdown.set()
        self.sock_client.close()
        self.sock_internal.close()
        logger.info("Shutdown complete.")

    def _run(self):
        """Main event loop using efficient I/O multiplexing."""
        while not self._shutdown.is_set():
            # Process worker messages (heartbeats, results)
            self._process_internal()
            
            # Process client messages (tasks)
            self._process_client()

    def _process_internal(self):
        """Handle messages from workers."""
        result = self.sock_internal.recv(timeout=0.001)
        if not result:
            return
            
        msg, addr = result
        self.metrics.messages_received += 1
        self.metrics.bytes_received += len(msg.payload)
        
        if msg.msg_type == MessageType.HEARTBEAT:
            self._handle_heartbeat(msg, addr)
        elif msg.msg_type == MessageType.RESULT:
            self._handle_result(msg, addr)
        elif msg.msg_type == MessageType.CHUNK:
            self._handle_chunk(msg, addr)
        elif msg.msg_type == MessageType.ERROR:
            self._handle_worker_error(msg, addr)

    def _process_client(self):
        """Handle messages from clients."""
        result = self.sock_client.recv(timeout=0.001)
        if not result:
            return
            
        msg, client_addr = result
        self.metrics.messages_received += 1
        self.metrics.bytes_received += len(msg.payload)
        
        if msg.msg_type == MessageType.TASK:
            self._handle_task(msg, client_addr)
        elif msg.msg_type == MessageType.HEALTH:
            self._handle_health_request(msg, client_addr)

    def _handle_heartbeat(self, msg: DCFMessage, addr: Tuple[str, int]):
        """Process worker heartbeat."""
        try:
            worker_port = int(msg.payload.decode('utf-8'))
            worker_addr = (addr[0], worker_port)
            self.registry.register(worker_addr)
        except (ValueError, UnicodeDecodeError) as e:
            logger.warning(f"Invalid heartbeat from {addr}: {e}")

    def _handle_result(self, msg: DCFMessage, addr: Tuple[str, int]):
        """Process inference result from worker."""
        request = self.tracker.complete(msg.sequence)
        if not request:
            logger.warning(f"Result {msg.sequence} has no pending request (timeout?)")
            return
        
        # Calculate latency
        latency_ms = (time.time() - request.created_at) * 1000
        
        # Update worker stats
        if request.worker_addr:
            self.registry.complete_task(request.worker_addr, latency_ms, success=True)
        
        # Forward to client
        response = DCFMessage.result(msg.sequence, msg.payload)
        self.sock_client.send_chunked(response, request.client_addr)
        
        self.metrics.messages_sent += 1
        self.metrics.bytes_sent += len(msg.payload)
        self.metrics.tasks_processed += 1
        self.metrics.record_latency(latency_ms)
        
        logger.info(f"✓ Task {msg.sequence} complete → {request.client_addr[0]} ({latency_ms:.0f}ms)")

    def _handle_chunk(self, msg: DCFMessage, addr: Tuple[str, int]):
        """Handle chunked result reassembly."""
        try:
            total, idx, checksum, data = parse_chunk(msg)
            
            if msg.sequence not in self._chunk_assemblers:
                self._chunk_assemblers[msg.sequence] = ChunkAssembler(
                    total_chunks=total,
                    checksum=checksum
                )
            
            assembler = self._chunk_assemblers[msg.sequence]
            if assembler.add_chunk(idx, data):
                # All chunks received
                complete_payload = assembler.assemble()
                del self._chunk_assemblers[msg.sequence]
                
                if complete_payload:
                    # Create synthetic result message
                    result_msg = DCFMessage.result(msg.sequence, complete_payload)
                    self._handle_result(result_msg, addr)
                else:
                    logger.error(f"Chunk assembly failed for {msg.sequence}")
                    
        except Exception as e:
            logger.error(f"Chunk processing error: {e}")

    def _handle_worker_error(self, msg: DCFMessage, addr: Tuple[str, int]):
        """Handle error from worker."""
        request = self.tracker.complete(msg.sequence)
        if request:
            if request.worker_addr:
                self.registry.complete_task(request.worker_addr, 0, success=False)
            
            # Forward error to client
            self.sock_client.send(msg, request.client_addr)
            self.metrics.tasks_failed += 1
            logger.warning(f"✗ Task {msg.sequence} failed on worker {addr}")

    def _handle_task(self, msg: DCFMessage, client_addr: Tuple[str, int]):
        """Route incoming task to a worker."""
        worker_addr = self.registry.get_worker(prefer_idle=True)
        
        if not worker_addr:
            error = DCFMessage.error(msg.sequence, ErrorCode.NO_WORKERS, "No workers available")
            self.sock_client.send(error, client_addr)
            logger.error(f"✗ No workers for task {msg.sequence}")
            return
        
        # Track the request
        if not self.tracker.add(msg.sequence, client_addr, worker_addr, len(msg.payload)):
            error = DCFMessage.error(msg.sequence, ErrorCode.WORKER_BUSY, "Server at capacity")
            self.sock_client.send(error, client_addr)
            return
        
        # Mark worker as busy and forward
        self.registry.assign_task(worker_addr, msg.sequence)
        
        fwd_msg = DCFMessage.task(msg.sequence, msg.payload)
        self.sock_internal.send(fwd_msg, worker_addr)
        
        self.metrics.messages_sent += 1
        self.metrics.bytes_sent += len(msg.payload)
        
        logger.info(f"→ Task {msg.sequence} routed to {worker_addr[0]}:{worker_addr[1]}")

    def _handle_health_request(self, msg: DCFMessage, client_addr: Tuple[str, int]):
        """Respond to health check."""
        stats = self.registry.get_stats()
        payload = json.dumps(stats).encode('utf-8')
        response = DCFMessage(
            msg_type=MessageType.HEALTH,
            sequence=msg.sequence,
            timestamp=DCFMessage.current_timestamp_micros(),
            payload=payload
        )
        self.sock_client.send(response, client_addr)

    def _maintenance_loop(self):
        """Background maintenance tasks."""
        while not self._shutdown.is_set():
            time.sleep(5.0)
            
            # Prune stale workers
            pruned_workers = self.registry.prune_stale()
            
            # Cleanup expired requests
            expired_requests = self.tracker.cleanup_expired()
            
            # Cleanup stale chunk assemblers
            stale_chunks = [seq for seq, asm in self._chunk_assemblers.items() 
                          if asm.is_stale()]
            for seq in stale_chunks:
                del self._chunk_assemblers[seq]
                logger.warning(f"Cleaned up stale chunk assembly for {seq}")
            
            if pruned_workers or expired_requests or stale_chunks:
                logger.debug(f"Maintenance: pruned {pruned_workers} workers, "
                           f"{expired_requests} requests, {len(stale_chunks)} chunks")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    controller = HeadController()
    
    # Signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        controller.shutdown()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        controller.start()
    except KeyboardInterrupt:
        controller.shutdown()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()

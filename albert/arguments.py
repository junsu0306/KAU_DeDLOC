# file: adaptive_strategy.py

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import cvxpy as cp

@dataclass
class PeerInfo:
    peer_id: str
    compute_speed: float  # samples per second
    upload: float         # upload bandwidth (Mbps)
    download: float       # download bandwidth (Mbps)
    pairwise_bw: Dict[str, float]  # peer_id -> bandwidth (Mbps)

class AdaptiveRoleMixin:
    def set_role(self, role: dict):
        self.role = role

    def get_send_targets(self) -> List[str]:
        return self.role.get("send_to", []) if self.role else []

    def get_recv_sources(self) -> List[str]:
        return self.role.get("recv_from", []) if self.role else []

    def should_send_to(self, peer_id: str) -> bool:
        return peer_id in self.get_send_targets()

    def should_receive_from(self, peer_id: str) -> bool:
        return peer_id in self.get_recv_sources()


def collect_peer_info(metrics_list) -> List[PeerInfo]:
    """
    Convert peer metrics (from DHT) into structured PeerInfo.
    """
    peer_info_list = []
    for peer_id, m in metrics_list.items():
        peer_info_list.append(
            PeerInfo(
                peer_id=peer_id,
                compute_speed=m["samples_per_second"],
                upload=m.get("upload_bw", 100.0),
                download=m.get("download_bw", 100.0),
                pairwise_bw=m.get("pairwise", {})  # optional
            )
        )
    return peer_info_list


def solve_optimal_plan(peers: List[PeerInfo]) -> Dict[str, dict]:
    """
    Solve LP based on DeDLOC paper (Equation 1) to assign roles.
    Extended to include send_to / recv_from mappings for gradient routing.
    """
    n = len(peers)
    peer_ids = [p.peer_id for p in peers]

    # Variables
    c = cp.Variable(n, boolean=True)  # compute role
    g = cp.Variable((n, n), boolean=True)  # g[i][j] == 1 if i aggregates for j

    # Build constraints
    constraints = []

    # Constraint: if i aggregates for j, then i must be a compute peer
    for i in range(n):
        for j in range(n):
            if i != j:
                constraints.append(g[i, j] <= c[i])

    # Constraint: limit aggregation load (heuristic)
    max_agg = n // 2
    for i in range(n):
        constraints.append(cp.sum(g[i, :]) <= max_agg)

    # Constraint: upload/download limits (simplified)
    for i in range(n):
        upload_limit = peers[i].upload
        download_limit = peers[i].download
        constraints.append(cp.sum(g[i, :]) <= upload_limit / 10)  # assume fixed gradient size
        constraints.append(cp.sum(g[:, i]) <= download_limit / 10)

    # Objective: maximize total throughput = sum of compute speeds of selected compute peers
    compute_speeds = np.array([p.compute_speed for p in peers])
    objective = cp.Maximize(cp.sum(cp.multiply(compute_speeds, c)))

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI)

    roles = {}
    c_val = c.value.astype(int)
    g_val = g.value.astype(int)

    for i, peer in enumerate(peers):
        aggregate_for = [peer_ids[j] for j in range(n) if g_val[i, j] == 1]
        send_to = [peer_ids[k] for k in range(n) if g_val[k, i] == 1]  # i sends to these peers
        recv_from = [peer_ids[k] for k in range(n) if g_val[i, k] == 1]  # i receives from these peers

        roles[peer.peer_id] = {
            "compute": bool(c_val[i]),
            "aggregate_for": aggregate_for,
            "send_to": send_to,
            "recv_from": recv_from
        }

    return roles


def download_my_role(dht, experiment_prefix: str, my_key: str) -> dict:
    roles_key = f"{experiment_prefix}_roles"
    roles = dht.get(roles_key, latest=True)
    if roles is not None:
        roles_dict = roles.value
        return roles_dict.get(my_key, {"compute": True, "aggregate_for": []})
    return {"compute": True, "aggregate_for": []}


def adaptive_aggregate(optimizer, role: dict):
    """
    Role-aware aggregation control at runtime.
    - If compute=False: skip gradient computation and aggregation.
    - If aggregate_for not empty: force aggregation step.
    """
    if not role.get("compute", True):
        return

    if role.get("aggregate_for"):
        if hasattr(optimizer, "averager"):
            optimizer.averager.schedule_averaging_step(group_key="adaptive", timeout=10.0)

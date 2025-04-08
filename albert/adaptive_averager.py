# file: adaptive_strategy.py

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import cvxpy as cp
import hashlib
import networkx as nx

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

    def get_group_key(self) -> str:
        if "subgroup_id" in self.role:
            return f"subgroup::{self.role['subgroup_id']}"
        elif "recv_from" in self.role:
            targets = ":".join(sorted(self.role["recv_from"]))
            return f"adaptive::{targets}"
        return "adaptive"


def collect_peer_info(metrics_list) -> List[PeerInfo]:
    peer_info_list = []
    for peer_id, m in metrics_list.items():
        peer_info_list.append(
            PeerInfo(
                peer_id=peer_id,
                compute_speed=m["samples_per_second"],
                upload=m.get("upload_bw", 100.0),
                download=m.get("download_bw", 100.0),
                pairwise_bw=m.get("pairwise", {})
            )
        )
    return peer_info_list

# Fault-tolerant enhancement: add backup fallback edges
def solve_optimal_plan(peers: List[PeerInfo]) -> Dict[str, dict]:
    n = len(peers)
    peer_ids = [p.peer_id for p in peers]
    c = cp.Variable(n, boolean=True)

    # Expand g[i,j] to allow multi-hop aggregation paths (soft routing graph)
    g = cp.Variable((n, n), boolean=True)

    constraints = []
    for i in range(n):
        for j in range(n):
            if i != j:
                constraints.append(g[i, j] <= c[i])

    max_agg = n // 2
    for i in range(n):
        constraints.append(cp.sum(g[i, :]) <= max_agg)
        constraints.append(cp.sum(g[i, :]) <= peers[i].upload / 10)
        constraints.append(cp.sum(g[:, i]) <= peers[i].download / 10)

    compute_speeds = np.array([p.compute_speed for p in peers])

    cost_matrix = np.zeros((n, n))
    for i, p_i in enumerate(peers):
        for j, p_j in enumerate(peers):
            if i != j:
                bw = p_i.pairwise_bw.get(p_j.peer_id, 100.0)
                cost_matrix[i, j] = 1.0 / max(bw, 1e-6)

    alpha = 0.1
    total_cost = cp.sum(cp.multiply(cost_matrix, g))
    objective = cp.Maximize(cp.sum(cp.multiply(compute_speeds, c)) - alpha * total_cost)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI)

    roles = {}
    c_val = c.value.astype(int)
    g_val = g.value.astype(int)

    # Build primary routing graph from g[i,j] matrix
    G = nx.DiGraph()
    G.add_nodes_from(peer_ids)
    for i in range(n):
        for j in range(n):
            if g_val[i, j] == 1:
                G.add_edge(peer_ids[i], peer_ids[j])

    # Add backup fallback edges to increase fault-tolerance
    for i in range(n):
        for j in range(n):
            if i != j and g_val[i, j] == 0:
                bw = peers[i].pairwise_bw.get(peers[j].peer_id, 0)
                if bw >= 50:  # threshold for usable backup connection
                    if not G.has_edge(peer_ids[i], peer_ids[j]):
                        G.add_edge(peer_ids[i], peer_ids[j])

    for peer in peer_ids:
        try:
            recv_from = list(nx.ancestors(G, peer))
            send_to = list(G.successors(peer))
            aggregate_for = list(G.predecessors(peer))
        except:
            recv_from, send_to, aggregate_for = [], [], []

        group_hash = hashlib.md5(":".join(sorted(recv_from)).encode()).hexdigest()[:8] if recv_from else "default"

        roles[peer] = {
            "compute": bool(c_val[peer_ids.index(peer)]),
            "aggregate_for": aggregate_for,
            "send_to": send_to,
            "recv_from": recv_from,
            "subgroup_id": group_hash
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
    if not role.get("compute", True):
        return
    if role.get("aggregate_for"):
        if hasattr(optimizer, "averager"):
            optimizer.averager.schedule_averaging_step(group_key="adaptive", timeout=10.0)

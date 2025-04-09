# adaptive_strategy.py

import numpy as np
import cvxpy as cp
from typing import List, Dict
from dataclasses import dataclass
import networkx as nx
import hashlib

@dataclass
class PeerInfo:
    """
    각 피어의 메트릭 정보를 담는 데이터 클래스.

    Attributes:
      peer_id: 피어의 식별자.
      compute_speed: 해당 피어의 연산 속도 (samples per second).
      upload: 업로드 대역폭 (Mbps).
      download: 다운로드 대역폭 (Mbps).
      pairwise_bw: 다른 피어와의 연결 대역폭
    """
    peer_id: str
    compute_speed: float
    upload: float
    download: float
    pairwise_bw: Dict[str, float]

def collect_peer_info(metrics_list) -> List[PeerInfo]:
    """
    DHT나 메트릭 시스템에서 얻은 피어 메트릭(dict 형식)을 PeerInfo 리스트로 변환.
    """
    peer_info_list = []
    for peer_id, m in metrics_list.items():
        peer_info_list.append(
            PeerInfo(
                peer_id=peer_id,
                compute_speed=m.get("samples_per_second", 100.0),
                upload=m.get("upload_bw", 100.0),
                download=m.get("download_bw", 100.0),
                pairwise_bw=m.get("pairwise", {})
            )
        )
    return peer_info_list

def solve_optimal_plan(peers: List[PeerInfo], B: float, P: float) -> Dict[str, dict]:
    """
    전체 시스템 throughput T는 다음과 같이 정의.
       T = min( (∑ s_i · c_i) / B,  min_{i}(∑_{j} g[j,i]) / P )
    여기서 목표는 T를 최대화.

    변수:
      - c[i]: 피어 i가 compute 역할을 수행하면 1, 아니면 0.
      - g[i, j]: 피어 i가 피어 j의 그라디언트 aggregation을 담당하면 1, 아니면 0.
      - T: 전체 시스템 throughput (두 측면의 병목 조건을 모두 반영)

    제약 조건:
      (1) 모든 i ≠ j에 대해, g[i, j] ≤ c[i]  
          (피어 i가 aggregation 역할을 할 때는 반드시 compute 역할이어야 함)
      (2) 각 피어의 aggregation 부하는 제한됨:
          - cp.sum(g[i, :]) ≤ max_agg (여기서 max_agg는 n//2로 임의 지정)
          - cp.sum(g[i, :]) ≤ peers[i].upload/10  
          - cp.sum(g[:, i]) ≤ peers[i].download/10
      (3) Compute throughput 제약:  
          T ≤ (∑ s_i · c[i]) / B
      (4) Aggregation throughput 제약:  
          ∀ i, T ≤ (∑_{j} g[j,i]) / P

    Parameters:
      - peers: 각 피어의 네트워크 및 연산 정보를 담은 PeerInfo 리스트.
      - B: target batch size 또는 scaling factor (예: 32)
      - P: Aggregation Scaling Factor (예: 1e5)
      
    Returns:
      각 피어의 역할 정보를 담은 dict (key는 peer_id), 예를 들어:
      { "peer1": {"compute": True, "aggregate_for": [...], "send_to": [...], "recv_from": [...], "subgroup_id": "abcd"}, ... }
    """
    n = len(peers)
    peer_ids = [p.peer_id for p in peers]
    
    # 결정 변수 선언
    c = cp.Variable(n, boolean=True)
    g = cp.Variable((n, n), boolean=True)
    T = cp.Variable(nonneg=True)  # Throughput 변수

    constraints = []
    
    # (1) g[i, j] ≤ c[i] (i != j)
    for i in range(n):
        for j in range(n):
            if i != j:
                constraints.append(g[i, j] <= c[i])
    
    # (2) 각 피어의 aggregation 부하 제한
    max_agg = n // 2  # 임의의 최대 aggregation 개수
    for i in range(n):
        constraints.append(cp.sum(g[i, :]) <= max_agg)
        constraints.append(cp.sum(g[i, :]) <= peers[i].upload / 10)
        constraints.append(cp.sum(g[:, i]) <= peers[i].download / 10)
    
    compute_speeds = np.array([p.compute_speed for p in peers])
    
    # (3) Compute throughput 제약: T ≤ (∑ s_i * c[i]) / B
    constraints.append(T <= cp.sum(cp.multiply(compute_speeds, c)) / B)
    
    # (4) Aggregation throughput 제약: 모든 i에 대해 T ≤ (∑_{j} g[j,i]) / P
    for i in range(n):
        constraints.append(T <= cp.sum(g[:, i]) / P)
    
    # 목표: T 최대화
    objective = cp.Maximize(T)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI)
    
    # 결정 변수의 값 평가
    c_val = c.value.astype(int)
    g_val = g.value.astype(int)
    
    # 네트워크 라우팅 그래프 구성: g의 결과를 기반으로 Directed Graph 구성
    G = nx.DiGraph()
    G.add_nodes_from(peer_ids)
    for i in range(n):
        for j in range(n):
            if i != j and g_val[i, j] == 1:
                G.add_edge(peer_ids[i], peer_ids[j])
    
    # 백업(backup) 경로 추가: 연결 대역폭이 일정 임계치(50 Mbps) 이상이면 추가
    for i in range(n):
        for j in range(n):
            if i != j and g_val[i, j] == 0:
                bw = peers[i].pairwise_bw.get(peer_ids[j], 0)
                if bw >= 50:
                    if not G.has_edge(peer_ids[i], peer_ids[j]):
                        G.add_edge(peer_ids[i], peer_ids[j])
    
    # 각 피어별 역할 정보를 구성하여 dict로 반환
    roles = {}
    for peer in peer_ids:
        try:
            recv_from = list(nx.ancestors(G, peer))
            send_to = list(G.successors(peer))
            aggregate_for = list(G.predecessors(peer))
        except Exception:
            recv_from, send_to, aggregate_for = [], [], []
        subgroup_id = hashlib.md5(":".join(sorted(recv_from)).encode()).hexdigest()[:8] if recv_from else "default"
        roles[peer] = {
            "compute": bool(c_val[peer_ids.index(peer)]),
            "aggregate_for": aggregate_for,
            "send_to": send_to,
            "recv_from": recv_from,
            "subgroup_id": subgroup_id
        }
    
    return roles

def download_my_role(dht, experiment_prefix: str, my_key: str) -> dict:
    """
    DHT에 저장된 역할 정보를 이용하여 현재 피어의 역할을 다운로드.
    
    역할 정보는 DHT에 '{experiment_prefix}_roles'라는 키로 저장되어 있고, 각 피어의 역할은
    자신의 식별자(my_key)를 key로 하여 존재. 만약 정보가 없으면, 기본적으로 compute 역할(True)
    및 빈 aggregate_for 목록을 반환.
    
    Parameters:
      dht: 피어-투-피어 통신을 위한 DHT 인스턴스.
      experiment_prefix: 역할 정보를 식별하기 위한 문자열 prefix.
      my_key: 현재 피어의 식별자로, 역할 정보를 조회할 때 사용됨.
    
    Returns:
      현재 피어의 역할 정보를 담은 dict.
    """
    roles_key = f"{experiment_prefix}_roles"
    roles = dht.get(roles_key, latest=True)
    if roles is not None:
        roles_dict = roles.value
        return roles_dict.get(my_key, {"compute": True, "aggregate_for": []})
    return {"compute": True, "aggregate_for": []}

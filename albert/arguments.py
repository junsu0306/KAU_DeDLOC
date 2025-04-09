# arguments.py

from transformers import TrainingArguments
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass, field
import cvxpy as cp

@dataclass
class BaseTrainingArguments:
    experiment_prefix: str = field(
        metadata={"help": "A unique 'name' of this experiment, used to store metadata on the DHT"}
    )
    initial_peers: List[str] = field(
        default_factory=list,
        metadata={"help": "One or more peers (comma-separated) that will welcome you into the collaboration"},
    )
    dht_listen_on: str = field(
        default="[::]:*", metadata={"help": "Network interface used for incoming DHT communication. Default: all ipv6"}
    )
    
@dataclass
class AveragerArguments:
    averaging_expiration: float = field(
        default=5.0, metadata={"help": "Averaging group will wait for stragglers for at most this many seconds"}
    )
    averaging_timeout: float = field(
        default=30.0, metadata={"help": "Give up on averaging step after this many seconds"}
    )
    listen_on: str = field(
        default="[::]:*",
        metadata={"help": "Network interface used for incoming averager communication. Default: all ipv6"},
    )
    min_refresh_period: float = field(
        default=0.5, metadata={"help": "Wait for at least this many seconds before fetching new collaboration state"}
    )
    max_refresh_period: float = field(
        default=30, metadata={"help": "Wait for at most this many seconds before fetching new collaboration state"}
    )
    default_refresh_period: float = field(
        default=3, metadata={"help": "Attempt to fetch collaboration state every this often until successful"}
    )
    expected_drift_peers: float = field(
        default=3, metadata={"help": "Trainer assumes that this many new peers can join per step"}
    )
    expected_drift_rate: float = field(
        default=0.2, metadata={"help": "Trainer assumes that this fraction of current size can join per step"}
    )
    performance_ema_alpha: float = field(
        default=0.1, metadata={"help": "Uses this alpha for moving average estimate of samples per second"}
    )
    target_group_size: int = field(default=256, metadata={"help": "Maximum group size for all-reduce"})
    metadata_expiration: float = field(
        default=30, metadata={"help": "Peer's metadata will be removed if not updated in this many seconds"}
    )


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
    DHT에서 얻은 피어 메트릭(dict 형식)을 PeerInfo 리스트로 변환.
    """
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

def solve_optimal_plan(peers: List[PeerInfo], B: float, P: float) -> Dict[str, dict]:
    """
    역할 할당 LP 문제를 풀어, 각 피어의 역할을 결정.
    Parameters:
      - peers: 각 피어의 정보를 담은 리스트.
      - B: target batch size 
      - P: Aggregation Scaling Factor 
    
    Returns:
      각 피어의 역할 정보를 담은 dict.
    """
    n = len(peers)
    peer_ids = [p.peer_id for p in peers]

    # 변수 선언: c, g, T (위와 동일)
    c = cp.Variable(n, boolean=True)
    g = cp.Variable((n, n), boolean=True)
    T = cp.Variable(nonneg=True)

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
    constraints.append(T <= cp.sum(cp.multiply(compute_speeds, c)) / B)
    for i in range(n):
        constraints.append(T <= cp.sum(g[:, i]) / P)

    objective = cp.Maximize(T)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI)

    c_val = c.value.astype(int)
    g_val = g.value.astype(int)
    roles = {}
    for i, peer in enumerate(peers):
        aggregate_for = [peer_ids[j] for j in range(n) if g_val[i, j] == 1]
        send_to = [peer_ids[k] for k in range(n) if g_val[k, i] == 1]
        recv_from = [peer_ids[k] for k in range(n) if g_val[i, k] == 1]
        roles[peer.peer_id] = {
            "compute": bool(c_val[i]),
            "aggregate_for": aggregate_for,
            "send_to": send_to,
            "recv_from": recv_from
        }
    return roles

def download_my_role(dht, experiment_prefix: str, my_key: str) -> dict:
    roles_key = f"{experiment_prefix}_roles"
    roles = dht.get(roles_key, latest=True)  #DHT에서 peer 메트릭을 수집
    if roles is not None:
        roles_dict = roles.value
        return roles_dict.get(my_key, {"compute": True, "aggregate_for": []})
    return {"compute": True, "aggregate_for": []}

@dataclass
class DatasetArguments:
    dataset_path: Optional[str] = field(
        default="data/bert_tiny_tokenized_wikitext103",  # 기존 wikitext-2 경로에서 수정
        metadata={"help": "Path to the tokenized dataset"},
    )

    tokenizer_path: Optional[str] = field(
        default="data/tokenizer_bert_tiny",  # tokenizer도 바꾼 경로로 맞춰주기
        metadata={"help": "Path to the tokenizer"},
    )

    config_path: Optional[str] = field(
        default="https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/main/config.json",
        metadata={"help": "Path to the model config"}
    )
    cache_dir: Optional[str] = field(default="data", metadata={"help": "Path to the cache"})



@dataclass
class BertTrainingArguments(TrainingArguments):
    dataloader_num_workers: int = 4
    per_device_train_batch_size: int = 128  # BERT-tiny는 작은 모델이므로 배치 크기 증가 가능
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1  # 작은 모델이므로 gradient accumulation을 낮춰도 됨
    seq_length: int = 128  # BERT-tiny는 짧은 문장에 최적화되어 있으므로 128로 조정

    max_steps: int = 1500_000  # 기존에서 더 늘림림
    learning_rate: float = 0.0008  # 기존 0.00176 → 0.0003으로 조정
    warmup_steps: int = 5000
    adam_epsilon: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    clamp_value: float = 10000.0

    fp16: bool = True  # 처음에는 False로 설정 후 문제 없으면 True로 변경
    fp16_opt_level: str = "O2"
    do_train: bool = True

    logging_steps: int = 10
    save_total_limit: int = 2
    save_steps: int = 500

    output_dir: str = "outputs"


    #Parital Staleness 추가
    partial_stale: bool = field(
        default=False,
        metadata={"help": "If True, uses a 1-step delayed gradient update for partial staleness."}
    )

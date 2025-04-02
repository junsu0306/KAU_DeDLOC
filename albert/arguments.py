from typing import Optional, List
from dataclasses import dataclass, field

from transformers import TrainingArguments


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
class CollaborativeOptimizerArguments:
    target_batch_size: int = field(
        default=4096,
        metadata={"help": "Perform optimizer step after all peers collectively accumulate this many samples"},
    )
    client_mode: bool = field(
        default=False,
        metadata={"help": "Of True, runs training without incoming connections, in a firewall-compatible mode"},
    )
    batch_size_lead: int = field(
        default=0,
        metadata={"help": "Optional: begin looking for group in advance, this many samples before target_batch_size"},
    )
    bandwidth: float = field(
        default=100.0,
        metadata={"help": "Available network bandwidth, in mbps (used for load balancing in all-reduce)"},
    )
    compression: str = field(
        default="FLOAT16", metadata={"help": "Use this compression when averaging parameters/gradients"}
    )


@dataclass
class CollaborationArguments(AveragerArguments, CollaborativeOptimizerArguments, BaseTrainingArguments):
    statistics_expiration: float = field(
        default=600, metadata={"help": "Statistics will be removed if not updated in this many seconds"}
    )
    endpoint: Optional[str] = field(
        default=None,
        metadata={"help": "This node's IP for inbound connections, used when running from behind a proxy"},
    )


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
    per_device_train_batch_size: int = 64  # BERT-tiny는 작은 모델이므로 배치 크기 증가 가능
    per_device_eval_batch_size: int = 2  # 기존 16
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
    do_eval: bool = True

    logging_steps: int = 10
    save_total_limit: int = 2
    save_steps: int = 500

    output_dir: str = "outputs"

    #Parital Staleness 추가
    partial_stale: bool = field(
        default=False,
        metadata={"help": "If True, uses a 1-step delayed gradient update for partial staleness."}
    )

    eval_accumulation_steps: Optional[int] = field(  # ✅ 여기 추가!
        default=8,
        metadata={"help": "Number of eval steps to accumulate before transferring to CPU"}
    )
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
        metadata={"help": "If True, runs training without incoming connections, in a firewall-compatible mode"},
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
    wandb_project: Optional[str] = field(default=None, metadata={"help": "WandB project name"})

@dataclass
class DatasetArguments:
    dataset_path: Optional[str] = field(
        default="data/bert_tiny_tokenized_wikitext103",
        metadata={"help": "Path to the tokenized dataset"},
    )
    tokenizer_path: Optional[str] = field(
        default="data/tokenizer_bert_tiny",
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
    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    seq_length: int = 128

    max_steps: int = 1500000
    learning_rate: float = 0.0008
    warmup_steps: int = 5000
    adam_epsilon: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    clamp_value: float = 10000.0

    fp16: bool = True
    fp16_opt_level: str = "O2"
    do_train: bool = True

    logging_steps: int = 10
    save_total_limit: int = 2
    save_steps: int = 500

    output_dir: str = "outputs"

    # Partial Staleness 옵션 (1-step delayed gradient update)
    partial_stale: bool = field(
        default=False,
        metadata={"help": "If True, uses a 1-step delayed gradient update for partial staleness."}
    )

    # Adam 옵티마이저용 베타 파라미터
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for the Adam optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for the Adam optimizer"}
    )
    # 초기 시드 값 (실험의 재현성을 위해)
    seed: int = field(
        default=42, metadata={"help": "Random seed for initialization"}
    )

    @property
    def device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# adaptive_averager.py
import asyncio
import time
import logging
import random
import math
from typing import Dict, List, Optional, Tuple, Any, Sequence
import torch
import cvxpy as cp
import numpy as np

# hivemind 컴포넌트 import
from hivemind import DHT, PeerID, P2PContext, P2P, Endpoint
from hivemind.dht.crypto import RSAPublicKey
from hivemind.optim.progress_tracker import PerformanceEMA
from hivemind.utils import get_logger, MSGPackSerializer, deserialize_torch_tensor, serialize_torch_tensor, MPFuture, get_dht_time, Endpoint
from hivemind.compression import deserialize_compressed_tensor, serialize_compressed_tensor, choose_compression, CompressionType

# arguments import
from arguments import AdaptiveAveragerArguments, NetworkingArguments, BertTrainingArguments

logger = get_logger(__name__)

# 간단한 피어 상태 정보 저장용
class PeerInfo:
    def __init__(self, peer_id: PeerID, endpoint: Optional[Endpoint], public_key: RSAPublicKey,
                 performance: float = 0.0, bandwidth_ul: float = 1.0, bandwidth_dl: float = 1.0,
                 is_client: bool = False):
        self.peer_id = peer_id
        self.endpoint = endpoint
        self.public_key = public_key
        self.s_i = performance # samples/sec
        self.u_i = max(1.0, bandwidth_ul) * 1e6 / 8 # Mbps to Bytes/sec (최소 1Mbps 가정)
        self.d_i = max(1.0, bandwidth_dl) * 1e6 / 8 # Mbps to Bytes/sec
        self.is_client = is_client # 클라이언트 모드 여부

class AdaptiveAverager:
    """
    Adaptive Averaging Algorithm을 직접 구현.
    hivemind.DHT와 P2P 네트워킹 인프라를 활용.
    Fault Tolerance를 위한 랜덤 그룹핑 및 단순 실패 처리 포함.
    """
    def __init__(self, *,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 dht: DHT,
                 p2p: P2P, # hivemind의 P2P 인스턴스 사용
                 prefix: str,
                 target_batch_size: int,
                 args: Any, # TrainingPipelineArguments 인스턴스 (모든 설정 포함)
                 param_names: Optional[List[str]] = None, # 평균화 대상 파라미터 이름 (선택적)
                 **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dht = dht
        self.p2p = p2p
        self.prefix = prefix
        self.target_batch_size = target_batch_size
        self.args = args # TrainingPipelineArguments 전체 저장
        self.adaptive_args: AdaptiveAveragerArguments = args
        self.net_args: NetworkingArguments = args
        self.training_args: BertTrainingArguments = args

        self.grad_compression_type = CompressionType.Value[self.adaptive_args.grad_compression] # Enum으로 변환
        self.state_sync_interval = self.adaptive_args.state_sync_interval
        self.last_state_sync_time = 0

        # 파라미터 및 그래디언트 관리
        self.parameters = list(model.parameters())
        self.param_names = param_names or [name for name, _ in model.named_parameters()]
        self.gradient_accumulators = [torch.zeros_like(p, device='cpu') for p in self.parameters if p.requires_grad]
        self.num_parameters = sum(p.numel() for p in self.gradient_accumulators)
        self.device = next(model.parameters()).device

        # 상태 변수
        self.local_samples_accumulated = 0
        self.steps_taken_accum = 0
        self.global_step = 0
        self.performance_tracker = PerformanceEMA(alpha=0.1)
        self.serializer = MSGPackSerializer

        # 피어 상태 및 LP
        self.peer_info: Dict[PeerID, PeerInfo] = {}
        self.last_state_refresh_time = 0
        self.last_lp_solution: Optional[Dict] = None
        self.averaging_context: Optional[P2PContext] = None
        self.averaging_in_progress = False
        self.background_tasks: List[asyncio.Task] = []

        # Fault Tolerance 관련
        self.active_peer_failures: Dict[PeerID, float] = {} # 피어별 실패율 추정치 (EMA)
        self.estimated_failure_rate: float = 0.0 # 전체 평균 실패율 추정치

        # Partial Staleness 관련
        self.stale_gradient_buffer: Optional[List[torch.Tensor]] = None
        self.current_gradients_for_averaging: Optional[List[torch.Tensor]] = None
        self.averaging_future: Optional[MPFuture] = None # 또는 asyncio.Future

        logger.info(f"AdaptiveAverager initialized for peer {self.p2p.peer_id.to_base58()}. Target batch size: {self.target_batch_size}")
        logger.info(f"Fault tolerance enabled: {self.adaptive_args.use_fault_tolerance}")
        if self.adaptive_args.use_fault_tolerance:
            logger.info(f"Target group size (m): {self.adaptive_args.target_group_size_m}, "
                        f"Min group size: {self.adaptive_args.min_group_size}, "
                        f"Group comm timeout: {self.adaptive_args.group_comm_timeout}s")

        # 비동기 작업 시작
        if asyncio.get_event_loop().is_running():
             asyncio.create_task(self._initialize_async())
        else:
             asyncio.run(self._initialize_async())


    async def _initialize_async(self):
         """ 비동기 초기화 (핸들러 등록, 메타데이터 발행 등) """
         await self._register_p2p_handlers()
         self.background_tasks.append(asyncio.create_task(self._periodically_store_metadata()))


    async def _register_p2p_handlers(self):
        """ P2P 요청을 처리할 핸들러들을 등록 """
        # XXX: 실제 hivemind의 P2P API 확인 및 에러 처리 강화 필요
        try:
            logger.info("Registering P2P handlers...")
            protocol_base = f"{self.prefix}/adaptive_averaging/v1" # 프로토콜 정의

            # 그래디언트 청크 수신 핸들러
            await self.p2p.add_unary_handler(
                protocol_base + "/handle_gradient_chunk", self._handle_gradient_chunk,
            )
            # 평균 결과 요청 핸들러
            await self.p2p.add_unary_handler(
                 protocol_base + "/request_averaged_chunk", self._handle_result_request,
            )
            # 상태 동기화 핸들러 (선택적)
            # await self.p2p.add_unary_handler(protocol_base + "/get_current_state", self._handle_get_state)

            logger.info("P2P handlers registered.")
        except Exception as e:
            logger.exception("Failed to register P2P handlers")


    def _get_local_performance(self) -> float:
        """ 로컬 성능 추정치 반환 """
        if self.adaptive_args.compute_samples_per_second is not None:
            return self.adaptive_args.compute_samples_per_second
        # steps_taken_accum이 0일 경우 대비
        return self.performance_tracker.samples_per_second


    async def _store_local_metadata(self):
        """ 로컬 피어의 상태 정보를 주기적으로 DHT에 게시 """
        # ... (이전 답변의 구현과 유사, get_dht_time() 사용) ...
        try:
            now = get_dht_time()
            local_perf = self._get_local_performance()
            bw_ul = self.adaptive_args.bandwidth_ul
            bw_dl = self.adaptive_args.bandwidth_dl

            # 로컬 P2P endpoint 가져오기 (주의: 변경될 수 있음)
            endpoint = self.p2p.endpoint

            metadata = {
                'endpoint': endpoint,
                's_i': local_perf,
                'u_i': bw_ul, # Mbps
                'd_i': bw_dl, # Mbps
                'timestamp': now,
                'is_client': self.net_args.client_mode,
                'schema_version': 1
            }
            key = self.prefix + "_metadata"
            subkey = self.dht.crypto.get_public_key(format='bytes')
            await self.dht.store(
                key=key, subkey=subkey, value=metadata,
                expiration_time=now + self.adaptive_args.metadata_expiration,
            )
            logger.debug(f"Stored local metadata to DHT.")
        except Exception as e:
            logger.error(f"Failed to store local metadata: {e}", exc_info=True)


    async def _periodically_store_metadata(self):
        """ 백그라운드에서 주기적으로 메타데이터 저장 """
        while True:
             await self._store_local_metadata()
             await asyncio.sleep(self.adaptive_args.metadata_expiration / 2)


    async def _refresh_peer_states(self) -> bool:
        """ DHT에서 활성 피어들의 상태 정보를 가져와 self.peer_info 업데이트 """
        # ... (이전 답변의 구현과 유사, await dht.get 사용) ...
        now = get_dht_time()
        if now - self.last_state_refresh_time < self.adaptive_args.adaptive_state_refresh_period:
            return False
        logger.info("Refreshing peer states from DHT...")
        updated_info: Dict[PeerID, PeerInfo] = {}
        try:
            key = self.prefix + "_metadata"
            all_metadata = await self.dht.get(key, latest=True)
            if all_metadata:
                 metadata_dict = all_metadata.value
                 for subkey_bytes, entry in metadata_dict.items():
                     if entry.expiration_time >= now:
                         try:
                             state_data = entry.value
                             public_key = self.dht.crypto.restore_public_key(subkey_bytes)
                             peer_id = self.dht.crypto.key_to_peer_id(public_key)
                             # ... (state_data 파싱 및 PeerInfo 생성 - 이전 답변 참조) ...
                             p_info = PeerInfo(...)
                             updated_info[peer_id] = p_info
                         except Exception as e:
                             logger.warning(...)
            # 로컬 정보 추가 (이전 답변 참조)
            local_info = PeerInfo(...)
            updated_info[self.p2p.peer_id] = local_info

            self.peer_info = updated_info
            logger.info(f"Refreshed info for {len(self.peer_info)} peers.")
            self.last_state_refresh_time = now
            return True
        except Exception as e:
             logger.error(f"Error refreshing peer states: {e}", exc_info=True)
             return False


    def _solve_lp(self) -> Optional[Dict]:
        """ CVXPY를 사용하여 Adaptive Averaging LP 문제를 풉니다. """
        # ... (이전 답변의 구현과 동일/유사) ...
        if not self.peer_info: return None
        peers = list(self.peer_info.values())
        # ... (LP 변수, 제약조건, 문제 풀이 - 이전 답변 참조) ...
        try:
             # ... problem.solve() ...
             if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                 # ... 결과 처리 (non-negative 보장 등) ...
                 solution = { ... }
                 return solution
             else: # 실패 처리
                 return None
        except Exception as e:
             logger.error(...)
             return None


    # === P2P 통신 핸들러 (상세 구현 필요) ===
    async def _handle_gradient_chunk(self, request: Any, context: P2PContext) -> Any:
        """ 다른 피어로부터 그래디언트 청크를 수신하고 저장/집계 """
        # XXX: 상세 구현 필요 (역직렬화, 압축 해제, 유효성 검증, 로컬 버퍼 저장)
        logger.debug(f"Received gradient chunk from {context.remote_peer_id.to_base58()} (placeholder)")
        # 임시 성공 응답
        return self.serializer.dumps({"status": "ok", "message": "chunk received (placeholder)"})

    async def _handle_result_request(self, request: Any, context: P2PContext) -> Any:
        """ 다른 피어로부터 평균화된 결과 청크 요청 처리 """
        # XXX: 상세 구현 필요 (요청 청크 식별, 계산 완료 확인, 결과 직렬화/압축, 응답)
        logger.debug(f"Received result request from {context.remote_peer_id.to_base58()} (placeholder)")
        return self.serializer.dumps({"status": "not_implemented", "data": None})


    # === Fault Tolerance 로직 ===
    def _calculate_num_rounds_and_group_size(self, n: int) -> Tuple[int, int]:
        """ 라운드 수(k)와 그룹 크기(m) 결정 """
        # ... (이전 답변의 구현과 동일/유사, 최적 m 계산 로직 추가 가능) ...
        if n <= self.adaptive_args.min_group_size: return 1, n
        m = min(n, self.adaptive_args.target_group_size_m)
        m = max(self.adaptive_args.min_group_size, m)
        k = math.ceil(math.log(n) / math.log(m)) if m > 1 and n > 0 else 1
        k = max(1, k)
        logger.info(f"Group averaging params: n={n}, target_m={self.adaptive_args.target_group_size_m} -> actual_m={m}, k={k} rounds")
        return k, m

    def _form_random_groups(self, active_peer_ids: List[PeerID], m: int) -> List[List[PeerID]]:
        """ 활성 피어들을 무작위로 크기 m의 그룹으로 나눔 """
        # ... (이전 답변의 구현과 동일/유사) ...
        if not active_peer_ids: return []
        n = len(active_peer_ids)
        shuffled_peers = random.sample(active_peer_ids, n)
        groups = []
        # ... (그룹 나누는 로직 - 이전 답변 참조) ...
        logger.debug(f"Formed {len(groups)} random groups.")
        return groups

    def _update_failure_rate(self, peer_ids: List[PeerID], success: bool):
         """ 피어 실패율 추정치 업데이트 (간단한 EMA) """
         # XXX: 상세 구현 필요
         # 예: alpha = self.adaptive_args.failure_rate_ema_alpha
         #     for pid in peer_ids:
         #         current_estimate = self.active_peer_failures.get(pid, 0.0)
         #         target = 0.0 if success else 1.0 # 또는 실패한 특정 피어만 1.0
         #         new_estimate = alpha * target + (1 - alpha) * current_estimate
         #         self.active_peer_failures[pid] = new_estimate
         #     self.estimated_failure_rate = np.mean(list(self.active_peer_failures.values())) if self.active_peer_failures else 0.0
         pass


    async def _perform_intra_group_averaging(self, gradients_to_average: List[torch.Tensor],
                                             group_peer_ids: List[PeerID], round_idx: int) -> Optional[List[torch.Tensor]]:
        """ 그룹 내에서 P2P 통신을 통해 평균화를 수행 (상세 구현 필요) """
        # ... (이전 답변의 구현과 유사, P2P 통신 상세 구현 필요) ...
        group_size = len(group_peer_ids)
        if group_size < self.adaptive_args.min_group_size: # 최소 그룹 크기 체크
            logger.warning(f"Group size {group_size} too small, skipping round {round_idx+1}.")
            return gradients_to_average # 현재 그래디언트 그대로 반환

        logger.info(f"Performing averaging in group (size {group_size}) for round {round_idx+1}")
        received_gradients: Dict[PeerID, List[torch.Tensor]] = {self.p2p.peer_id: gradients_to_average}
        active_responses = 1

        # XXX: 실제 P2P 통신(그래디언트 교환) 및 타임아웃 처리 구현 필요
        async def request_from_peer(peer_id: PeerID):
            nonlocal active_responses
            try:
                logger.debug(f"Requesting round {round_idx+1} gradients from {peer_id.to_base58()[-6:]}")
                # P2P call to get gradients (placeholder)
                # response = await asyncio.wait_for(
                #     self.p2p.call_unary_handler(peer_id, "_handle_get_round_gradient", ...),
                #     timeout=self.adaptive_args.group_comm_timeout
                # )
                # gradients = deserialize... (response)
                # received_gradients[peer_id] = gradients

                # 시뮬레이션
                await asyncio.sleep(0.1 + random.random() * 0.2) # 임시 지연
                if random.random() > 0.1: # 10% 실패 시뮬레이션
                     received_gradients[peer_id] = gradients_to_average # 임시: 자기 데이터 사용
                     active_responses += 1
                     logger.debug(f"Successfully received from {peer_id.to_base58()[-6:]}")
                else:
                     logger.warning(f"Simulated failure from peer {peer_id.to_base58()[-6:]}")

            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for round {round_idx+1} gradients from {peer_id.to_base58()[-6:]}")
            except Exception as e:
                logger.warning(f"Error communicating with peer {peer_id.to_base58()[-6:]}: {e}")

        tasks = [request_from_peer(pid) for pid in group_peer_ids if pid != self.p2p.peer_id]
        if tasks: await asyncio.gather(*tasks)

        if active_responses < self.adaptive_args.min_group_size:
             logger.warning(f"Only {active_responses} responses in group round {round_idx+1}. Averaging failed.")
             return None # 실패

        # 응답 온 피어들만으로 평균 계산 (단순 실패 처리)
        logger.info(f"Averaging round {round_idx+1} gradients from {active_responses} responsive peers.")
        final_gradients = [torch.zeros_like(g) for g in gradients_to_average]
        for peer_id, grads in received_gradients.items(): # 자기 자신 포함
             for i in range(len(final_gradients)):
                 final_gradients[i].add_(grads[i])
        for i in range(len(final_gradients)):
            final_gradients[i].div_(active_responses)

        return final_gradients


    # === 평균화 오케스트레이션 ===
    async def _orchestrate_averaging(self, gradients_cpu: List[torch.Tensor]) -> Optional[List[torch.Tensor]]:
        """ LP 솔루션 및 Fault Tolerance 설정에 따라 그래디언트 평균화 실행 """
        # ... (averaging_in_progress 체크, LP 솔루션/피어 정보 확인 - 이전 답변과 유사) ...
        if not self.last_lp_solution or not self.peer_info: return None
        self.averaging_in_progress = True
        # ... (활성 피어 ID 목록 `active_peer_ids` 생성) ...
        active_peer_ids = list(self.peer_info.keys())
        n = len(active_peer_ids)

        averaged_gradients = [g.clone() for g in gradients_cpu] # 최종 결과 저장용

        try:
            if not self.adaptive_args.use_fault_tolerance or n < self.adaptive_args.min_group_size * 2 :
                # 전역 평균화 (상세 구현 필요)
                logger.info("Performing global averaging (FT disabled or too few peers) - Placeholder!")
                await asyncio.sleep(0.5) # 임시
                # averaged_gradients = await _perform_global_p2p_averaging(...)
            else:
                # 그룹 평균화
                k, m = self._calculate_num_rounds_and_group_size(n)
                logger.info(f"Starting {k} rounds of group averaging with group size ~{m}.")
                current_round_gradients = averaged_gradients

                for round_idx in range(k):
                    # ... (그룹 형성, 내 그룹 찾기 - 이전 답변 참조) ...
                    groups = self._form_random_groups(active_peer_ids, m)
                    my_group_peer_ids = None
                    # ... (내 그룹 찾기 로직) ...
                    if not my_group_peer_ids: continue

                    # 그룹 내 평균화 실행
                    group_result = await self._perform_intra_group_averaging(
                        current_round_gradients, my_group_peer_ids, round_idx
                    )

                    # 실패 처리 및 결과 업데이트
                    if group_result is None:
                         logger.warning(f"Group averaging failed for round {round_idx+1}. Using previous results.")
                         self._update_failure_rate(my_group_peer_ids, success=False)
                         # 실패 시 현재 라운드 그래디언트 유지
                    else:
                         current_round_gradients = group_result # 성공 시 결과 업데이트
                         self._update_failure_rate(my_group_peer_ids, success=True)

                averaged_gradients = current_round_gradients # 최종 결과

        # ... (오류 처리 및 finally 블록 - 이전 답변과 유사) ...
        finally:
            self.averaging_in_progress = False
            # ...
            return averaged_gradients


    # === 옵티마이저 스텝 ===
    def step(self, loss: Optional[torch.Tensor] = None, batch_size: Optional[int] = None, **kwargs):
        """ 옵티마이저 스텝 함수 (훈련 루프에서 호출) """
        step_start_time = time.time()
        performed_optimizer_step = False

        # 1. 로컬 그래디언트 누적 (CPU accumulator 사용)
        if loss is not None and batch_size is not None and not self.adaptive_args.is_auxiliary:
            # ... (그래디언트 누적 로직 - 이전 답변과 동일) ...
            self.local_samples_accumulated += batch_size
            self.steps_taken_accum += 1
            # ... (성능 추적기 업데이트)

        # 2. 평균화 및 옵티마이저 스텝 조건 확인
        if (not self.adaptive_args.is_auxiliary and
                not self.averaging_in_progress and
                self.local_samples_accumulated >= self.target_batch_size):

            logger.info(f"Target batch size reached ({self.local_samples_accumulated}). Preparing averaging round {self.global_step + 1}.")
            self.current_gradients_for_averaging = [acc.clone() for acc in self.gradient_accumulators]
            for acc in self.gradient_accumulators: acc.zero_()
            current_samples = self.local_samples_accumulated
            self.local_samples_accumulated = 0
            self.steps_taken_accum = 0

            gradients_to_apply = None
            try:
                if self.training_args.partial_stale:
                    # 이전 결과 적용
                    if self.stale_gradient_buffer is not None:
                        gradients_to_apply = self.stale_gradient_buffer
                        logger.debug("Applying gradients from previous averaging round.")
                    else: logger.debug("No previous gradients to apply yet.")

                    # 현재 그래디언트 평균화는 백그라운드에서 시작 (다음 스텝용)
                    # XXX: 실제 비동기 실행 및 Future 관리 구현 필요
                    async def run_avg_and_store():
                        await self._refresh_peer_states() # 상태 먼저 갱신
                        self._solve_lp() # LP 풀고
                        result = await self._orchestrate_averaging(self.current_gradients_for_averaging)
                        self.stale_gradient_buffer = result # 결과 저장 (성공/실패 무관? 또는 실패 시 None?)
                        logger.info("Async averaging for next step completed (or failed).")

                    if not self.averaging_in_progress: # 중복 실행 방지
                       asyncio.create_task(run_avg_and_store())
                    else:
                       logger.warning("Previous async averaging seems to be running.")

                else: # Partial staleness 비활성화
                    # 현재 그래디언트 평균화 즉시 실행 (또는 대기)
                    logger.debug("Running averaging synchronously for current step.")
                    # 상태 갱신 및 LP 풀이는 _orchestrate_averaging 내부에서? -> 호출 전에 수행
                    async def run_sync_avg():
                         await self._refresh_peer_states()
                         self._solve_lp()
                         return await self._orchestrate_averaging(self.current_gradients_for_averaging)

                    # 이벤트 루프에서 실행 필요
                    # gradients_to_apply = asyncio.run(run_sync_avg()) # 스텝 함수 자체가 async가 아니면 문제 발생 가능
                    # 임시: 플레이스홀더 - 바로 위에서 정의한 비동기 함수를 이벤트 루프에서 실행하고 결과 기다림
                    try:
                         loop = asyncio.get_event_loop()
                         if loop.is_running():
                              # 이미 실행 중인 루프 사용 불가 -> 새 스레드에서 실행? 복잡함
                              logger.error("Cannot run async averaging synchronously inside a sync step function easily.")
                              gradients_to_apply = None # 임시 실패 처리
                         else:
                              gradients_to_apply = loop.run_until_complete(run_sync_avg())
                    except RuntimeError: # 루프가 없거나 이미 닫힌 경우 등
                         logger.error("Failed to get or run asyncio event loop for synchronous averaging.")
                         gradients_to_apply = None


                # 옵티마이저 스텝 적용
                if gradients_to_apply is not None:
                    self.optimizer.zero_grad()
                    with torch.no_grad():
                         for param, avg_grad in zip(self.parameters, gradients_to_apply):
                              if param.requires_grad: param.grad = avg_grad.to(self.device)
                    self.optimizer.step()
                    if self.scheduler: self.scheduler.step()
                    self.global_step += 1
                    performed_optimizer_step = True
                    logger.info(f"Optimizer step {self.global_step} completed.")
                    self._check_and_sync_state() # 상태 동기화 체크
                else:
                    logger.warning("No gradients to apply this step.")

            except Exception as e:
                logger.exception("Error occurred during the step execution")
                # 오류 발생 시 누적된 그래디언트/샘플 복구 또는 초기화 필요
                self.local_samples_accumulated = current_samples

        return performed_optimizer_step


    def load_state_from_peers(self):
        """ 다른 피어로부터 최신 모델/옵티마이저 상태 로드 시도 """
        # XXX: 상세 구현 필요
        logger.warning("load_state_from_peers is not implemented!")

    # 기타 필요한 함수들 (_check_and_sync_state, P2P 상태 요청 핸들러 등)
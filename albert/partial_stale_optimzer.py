# file: partial_stale_collaborative.py

import logging
import torch
from hivemind.optim.collaborative import CollaborativeOptimizer as BaseCollaborativeOptimizer

logger = logging.getLogger(__name__)

class PartialStaleCollaborativeOptimizer(BaseCollaborativeOptimizer):
    """
    1-step delayed update(Partial Staleness) 버전을 구현.
    iteration N에서 계산된 Gradient는 apply 안 하고,
    iteration N+1에서 apply하도록.

    핵심: step()에서 averaging → 기존 iteration gradient를 buffer에 넣기,
         이전 iteration buffer가 있다면 그걸 지금 apply
    """
    def __init__(self, partial_stale=False, *args, **kwargs):
        kwargs.pop("bandwidth", None)
         # fix: ensure 'start' key is present
        if 'start' not in kwargs:
            kwargs['start'] = True
        super().__init__(*args, **kwargs)
        self.partial_stale = partial_stale
        self.stale_grad_buffer = None  # 이전 iteration에서의 averaged gradient 저장
        
    def step(self, batch_size: int = None, **kwargs):
        """
        override: collaborative.py의 CollaborativeOptimizer.step(...)
        원래는 averaging+optimizer step을 즉시 수행.
        지금은 partial_stale=True면, 'apply'를 한 iteration 늦추기.
        """
        if not self.partial_stale:
            # 평소대로
            return super().step(batch_size=batch_size, **kwargs)

        # ============ partial_stale=True => 1-step delay ============
        # 1) 현재 iteration의 gradient averaging + local progress 업데이트 등은
        #    super().step(...)을 실행하되, 최종 apply 부분만 가로챈다.
        #    monkey-patch _apply_and_reset() or similar approach
        #    여기선 super().step() 안에서 "apply_accumulated_grads_" -> "opt.step()" 순으로 수행.
        #    -> 우리가 apply 부분만 intercept하면, buffer에 저장만 할 수 있다.

        # Trick: patch self.apply_accumulated_grads_ to store in buffer (instead of applying).
        orig_apply_accum = self.apply_accumulated_grads_
        local_grads = [None]  # mutable closure

        def store_in_buffer(scale_by=None):
            """
            replace the original function: just store the final grads in local_grads, do not opt.step()
            """
            # let's gather the final grads from accumulators (like done in the original function)
            # code snippet from the original apply_accumulated_grads_:
            # but we do NOT call self.opt.step() here
            param_list = []
            for group in self.opt.param_groups:
                param_list.extend(group["params"])
            
            grads = []
            if self.reuse_grad_buffers:
                # user is reusing param.grad as accum buffers => just copy them
                for p in param_list:
                    if p.grad is None:
                        grads.append(None)
                    else:
                        grads.append(p.grad.clone())
            else:
                # if we are using self._grads
                if self._grads is None:
                    # it might be lazily initalized
                    self._grads = [torch.zeros_like(p) for p in param_list]
                # scale if needed
                if scale_by is not None:
                    for g in self._grads:
                        g.mul_(scale_by)
                # now copy
                for g in self._grads:
                    grads.append(g.clone())

            local_grads[0] = grads

            # do not call self.opt.step() => we skip the actual update
            # do not zero out accumulators => keep them
            return
        
        # monkey-patch
        self.apply_accumulated_grads_ = store_in_buffer

        # call super step => does local progress, averaging, eventually calls apply_accumulated_grads_ (our patch)
        super().step(batch_size=batch_size, **kwargs)

        # revert patch
        self.apply_accumulated_grads_ = orig_apply_accum

        # 2) 이제 local_grads[0]에 "이번 iteration averaged gradient"가 들어있음
        #    이전 iteration buffer가 있으면 그걸 지금 apply
        if self.stale_grad_buffer is not None:
            self._apply_stale_grad(self.stale_grad_buffer)

        # 3) 이번 iteration gradient를 buffer에 저장
        if local_grads[0] is not None:
            # store for next iteration
            self.stale_grad_buffer = local_grads[0]
        else:
            logger.debug("No grad from the super step. Possibly no peers or local step was skipped?")
        
        return  # end of partial-stale step

    def _apply_stale_grad(self, grad_list):
        """
        실제로 param.grad에 grad_list를 복사한 뒤, self.opt.step() 호출
        """
        # grad_list는 list of Tensors (dense), one for each param
        param_list = []
        for group in self.opt.param_groups:
            param_list.extend(group["params"])
        
        if len(param_list) != len(grad_list):
            logger.warning("Mismatch: param_list len != grad_list len. Possibly a shape mismatch.")
        
        for p, g in zip(param_list, grad_list):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

        self.opt.step()

        # zero out param.grad
        for p in param_list:
            if p.grad is not None:
                p.grad = None

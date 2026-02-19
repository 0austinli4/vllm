# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.logprobs import (
    PromptLogprobs,
    SampleLogprobs,
    append_logprobs_for_next_position,
    create_prompt_logprobs,
    create_sample_logprobs,
)
from vllm.tokenizers.detokenizer_utils import (
    TokenizerLike,
    convert_ids_list_to_tokens,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

logger = init_logger(__name__)

NONES = itertools.repeat(None)


@dataclass
class LogprobsProcessor:
    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: TokenizerLike | None

    # Logprobs for this request
    logprobs: SampleLogprobs | None
    prompt_logprobs: PromptLogprobs | None
    cumulative_logprob: float | None
    num_logprobs: int | None
    num_prompt_logprobs: int | None

    # Confidence tracking (used for probe gate and history)
    conf_method: str = "window"  # "window" or "ema"
    conf_window_size: int = 120
    conf_ema_alpha: float = 0.01
    conf_ema_value: float = 0.0
    conf_min_tokens: int = 50
    conf_threshold: float = 12.0
    conf_topk: int = 10
    conf_rise_ratio: float = 1.5
    conf_samples: deque = field(default_factory=deque)
    conf_total_tokens: int = 0
    conf_cumulative_sum: float = 0.0
    conf_cumulative_count: int = 0
    conf_history: list = field(default_factory=list)  # [(total_tokens, avg_confidence)]

    # Probe-based stability config
    probe_enabled: bool = True
    probe_gate_enabled: bool = False   # default off → probe every 24 tokens
    probe_gate_threshold: float = 12.0
    probe_gate_rise_ratio: float = 1.5
    probe_gate_open: bool = False      # latches to True once gate opens
    probe_stability_n: int = 2
    slo_latency_ms: float | None = None  # None = SLO-aware mode disabled
    slo_base_k: int = 2                  # ceiling K; set from probe_stability_n at init
    probe_extraction_prompt: str = "\n\nFinal Answer:\n\\boxed{"
    probe_extraction_stop: str = "}"
    probe_extraction_max_tokens: int = 150
    # Output token counter — incremented from new_token_ids, independent of logprobs
    output_token_count: int = 0
    # Runtime probe state
    probe_history: list = field(default_factory=list)  # list[str | None]
    probe_pending: bool = False          # probe in-flight; don't double-spawn
    probe_should_stop: bool = False      # stability achieved
    probe_stable_answer: str | None = None

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        sampling_params = request.sampling_params
        assert sampling_params is not None
        num_logprobs = sampling_params.logprobs
        num_prompt_logprobs = sampling_params.prompt_logprobs

        # Extract probe and confidence parameters from extra_args
        extra_args = sampling_params.extra_args or {}
        probe_enabled      = extra_args.get("enable_answer_stability", True)
        probe_gate_enabled = extra_args.get("probe_gating", False)
        probe_gate_threshold   = extra_args.get("probe_gate_threshold", 12.0)
        probe_gate_rise_ratio  = extra_args.get("probe_gate_rise_ratio", 1.5)
        probe_stability_n      = extra_args.get("probe_stability_n", 2)
        slo_latency_ms         = extra_args.get("slo_latency_ms", None)
        probe_extraction_prompt     = extra_args.get("probe_extraction_prompt",
                                                     "\n\nFinal Answer:\n\\boxed{")
        probe_extraction_stop       = extra_args.get("probe_extraction_stop", "}")
        probe_extraction_max_tokens = extra_args.get("probe_extraction_max_tokens", 150)
        # Keep confidence params for gate and history recording
        conf_method      = extra_args.get("conf_method", "window")
        conf_window_size = extra_args.get("window_size", 120)
        conf_ema_alpha   = extra_args.get("conf_ema_alpha", 0.01)
        conf_min_tokens  = extra_args.get("min_tokens", 50)
        conf_threshold   = extra_args.get("conf_threshold", 12.0)
        conf_topk        = extra_args.get("conf_topk", 10)
        conf_rise_ratio  = extra_args.get("conf_rise_ratio", 1.5)

        # Validate logprobs >= conf_topk only if gate needs confidence
        if probe_enabled and probe_gate_enabled:
            if num_logprobs is None or num_logprobs < conf_topk:
                raise ValueError(
                    f"Probe gating requires logprobs >= conf_topk. "
                    f"Got logprobs={num_logprobs}, conf_topk={conf_topk}."
                )

        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.0),
            logprobs=(
                None
                if num_logprobs is None
                else create_sample_logprobs(sampling_params.flat_logprobs)
            ),
            prompt_logprobs=(
                None
                if num_prompt_logprobs is None
                else create_prompt_logprobs(sampling_params.flat_logprobs)
            ),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
            conf_method=conf_method,
            conf_window_size=conf_window_size,
            conf_ema_alpha=conf_ema_alpha,
            conf_ema_value=0.0,
            conf_min_tokens=conf_min_tokens,
            conf_threshold=conf_threshold,
            conf_topk=conf_topk,
            conf_rise_ratio=conf_rise_ratio,
            conf_samples=deque(),
            conf_total_tokens=0,
            conf_cumulative_sum=0.0,
            conf_cumulative_count=0,
            conf_history=[],
            probe_enabled=probe_enabled,
            probe_gate_enabled=probe_gate_enabled,
            probe_gate_threshold=probe_gate_threshold,
            probe_gate_rise_ratio=probe_gate_rise_ratio,
            probe_gate_open=False,
            probe_stability_n=probe_stability_n,
            slo_latency_ms=slo_latency_ms,
            slo_base_k=probe_stability_n,
            probe_extraction_prompt=probe_extraction_prompt,
            probe_extraction_stop=probe_extraction_stop,
            probe_extraction_max_tokens=probe_extraction_max_tokens,
            output_token_count=0,
            probe_history=[],
            probe_pending=False,
            probe_should_stop=False,
            probe_stable_answer=None,
        )

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """

        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists

        for rank_np, logprobs_np, token_ids_np in zip(
            ranks_lst, logprobs_lst, token_ids_lst
        ):
            rank = rank_np.tolist()
            logprobs = logprobs_np.tolist()
            token_ids = token_ids_np.tolist()
            # Detokenize (non-incrementally).
            decoded_tokens: list[str] | Iterable[None]
            if self.tokenizer is None:
                decoded_tokens = NONES
            else:
                decoded_tokens_list = convert_ids_list_to_tokens(
                    self.tokenizer, token_ids
                )
                decoded_tokens = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_list, tokens=token_ids
                )

            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]
            self.cumulative_logprob += sampled_token_logprob

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.logprobs,
                token_ids,
                logprobs,
                decoded_tokens,
                rank,
                self.num_logprobs,
            )

            # Update confidence tracking for gate and history
            if self.probe_enabled:
                self._update_confidence_sample(logprobs)

    def _update_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.

        """

        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        token_ids, logprobs, ranks = prompt_logprobs_tensors

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        all_decoded_tokens: list[str] | None = (
            None
            if self.tokenizer is None
            else convert_ids_list_to_tokens(
                self.tokenizer, token_ids.flatten().tolist()
            )
        )

        # Pythonize the torch tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids_list = token_ids.tolist()

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening and UTF-8 correction per position
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs

            decoded_tokens_for_pos: list[str] | Iterable[None]
            if all_decoded_tokens is None:
                decoded_tokens_for_pos = NONES
            else:
                # Extract decoded tokens for this position
                decoded_tokens_slice = all_decoded_tokens[offset:offset_end]
                # Apply UTF-8 correction within this position's token boundaries
                decoded_tokens_for_pos = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_slice, tokens=token_ids_list[pos]
                )

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.prompt_logprobs,
                token_ids_list[pos],
                prompt_logprobs[pos],
                decoded_tokens_for_pos,
                prompt_token_ranks[pos],
                self.num_prompt_logprobs,
            )

    def pop_prompt_logprobs(self) -> PromptLogprobs | None:
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = []
        return plp

    def _correct_decoded_token(self, idx: int, tokens: list[int]) -> str:
        assert self.tokenizer is not None, "self.tokenizer should not be None"

        # try with prev token id in same list
        if idx > 0:
            possible_decoded_token = self.tokenizer.decode(tokens[idx - 1 : idx + 1])
            if not possible_decoded_token.endswith("�"):
                return possible_decoded_token
        # try with previous logprob token id
        if self.logprobs:
            latest_token_id = next(iter(self.logprobs[-1]))

            decode_ids = [latest_token_id]
            if idx > 0:
                decode_ids.extend(tokens[idx - 1 : idx + 1])
            else:
                decode_ids.extend(tokens[idx : idx + 1])

            possible_decoded_token = self.tokenizer.decode(decode_ids)
            if not possible_decoded_token.endswith("�"):
                return possible_decoded_token

        # by default return empty string
        return ""

    def _verify_tokens(
        self, decoded_tokens_list: list[str], tokens: list[int]
    ) -> list[str]:
        corrected_decoded_token_map = dict()
        for idx, text in enumerate(decoded_tokens_list):
            if text.endswith("�"):
                # utf-8 char at the end means it's a potential unfinished byte sequence
                # from byte fallback tokenization.
                corrected_decoded_token_map[idx] = self._correct_decoded_token(
                    idx, tokens
                )

        for idx, text in corrected_decoded_token_map.items():
            decoded_tokens_list[idx] = text

        return decoded_tokens_list

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)

    def _update_confidence_sample(self, logprobs: list[float]) -> None:
        """Update confidence tracking with a new token's logprobs.

        Two methods:
        - "window": sliding window average over last conf_window_size tokens
        - "ema": exponential moving average, conf = (1-alpha)*conf + alpha*new

        Args:
            logprobs: List of top-k logprobs for this token position,
                     sorted by rank (sampled token first at index 0).
        """
        self.conf_total_tokens += 1

        # Compute confidence as negative mean of top-k logprobs
        topk_logprobs = logprobs[: self.conf_topk]
        confidence = -sum(topk_logprobs) / len(topk_logprobs)

        # Track cumulative stats for rise-over-mean
        self.conf_cumulative_sum += confidence
        self.conf_cumulative_count += 1

        if self.conf_method == "ema":
            # EMA: updated every token
            if self.conf_total_tokens == 1:
                self.conf_ema_value = confidence
            else:
                self.conf_ema_value = ((1 - self.conf_ema_alpha) * self.conf_ema_value
                                       + self.conf_ema_alpha * confidence)

            # Record history every 24 tokens
            if self.conf_total_tokens % 24 == 0:
                self.conf_history.append((self.conf_total_tokens, self.conf_ema_value))
        else:
            # Sliding window method
            self.conf_samples.append(confidence)
            if len(self.conf_samples) > self.conf_window_size:
                self.conf_samples.popleft()

            # Record history every 24 tokens
            if self.conf_total_tokens % 24 == 0:
                avg = sum(self.conf_samples) / len(self.conf_samples)
                self.conf_history.append((self.conf_total_tokens, avg))

    def _gate_open(self) -> bool:
        """Return True if the confidence gate has triggered (and latch it)."""
        if self.probe_gate_open:
            return True
        if self.conf_total_tokens < self.conf_min_tokens:
            return False
        if self.conf_method == "ema":
            current = self.conf_ema_value
        else:
            if not self.conf_samples:
                return False
            current = sum(self.conf_samples) / len(self.conf_samples)
        threshold_met = current > self.probe_gate_threshold
        cumulative_avg = (self.conf_cumulative_sum / self.conf_cumulative_count
                          if self.conf_cumulative_count else 0.0)
        rise_met = (cumulative_avg > 0 and
                    current / cumulative_avg >= self.probe_gate_rise_ratio)
        # Directional check for window mode
        if self.conf_method != "ema" and len(self.conf_samples) >= 2:
            if self.conf_samples[-1] < self.conf_samples[0]:
                return False
        if threshold_met or rise_met:
            self.probe_gate_open = True
        return self.probe_gate_open

    def increment_output_tokens(self, n: int) -> None:
        """Advance the output token counter by n. Called every decode step,
        independent of whether logprobs are computed."""
        if self.output_token_count == 0 and n > 0:
            logger.debug("PROBE: first token counted, probe_enabled=%s min_tokens=%d",
                         self.probe_enabled, self.conf_min_tokens)
        self.output_token_count += n

    def check_probe_trigger(self) -> bool:
        """Return True if a probe should be spawned this step."""
        if not self.probe_enabled:
            logger.debug("PROBE_TRIGGER: blocked — probe_enabled=False")
            return False
        if self.probe_pending or self.probe_should_stop:
            return False
        if self.output_token_count < self.conf_min_tokens:
            if self.output_token_count % 24 == 0 and self.output_token_count > 0:
                logger.debug("PROBE_TRIGGER: waiting for min_tokens (count=%d min=%d)",
                             self.output_token_count, self.conf_min_tokens)
            return False
        if self.output_token_count % 24 != 0:
            return False
        logger.debug("PROBE_TRIGGER: FIRE at output_token_count=%d", self.output_token_count)
        return True

    def mark_probe_pending(self) -> None:
        self.probe_pending = True

    def record_probe_result(self, answer: str | None) -> None:
        """Record a probe answer; set probe_should_stop if stable."""
        self.probe_pending = False
        self.probe_history.append(answer)
        logger.debug("PROBE_RESULT: answer=%r history_len=%d stability_n=%d",
                     answer, len(self.probe_history), self.probe_stability_n)
        if answer is None:
            return
        # Check last N consecutive matches
        n = self.probe_stability_n
        if len(self.probe_history) >= n:
            recent = self.probe_history[-n:]
            if all(r == answer for r in recent):
                self.probe_should_stop = True
                self.probe_stable_answer = answer
                logger.debug("PROBE_STABLE: answer=%r after %d probes", answer, len(self.probe_history))

    def slo_adjusted_k(self, elapsed_ms: float) -> int:
        """Return K adjusted according to SLO budget fraction consumed.

        Uses a continuous linear mapping:
            k(B) = k_max - (k_max - k_min) * B
        where B = elapsed / slo_latency (clamped to [0, 1]).

        k_max = slo_base_k (submitted probe_stability_n, ceiling).
        k_min = 2 (minimum useful stability window).

        As more of the SLO budget is consumed the required window shrinks,
        making early exit more aggressive.  If slo_latency_ms is None,
        SLO-aware mode is disabled and probe_stability_n is returned unchanged.
        """
        if self.slo_latency_ms is None or self.slo_latency_ms <= 0:
            return self.probe_stability_n
        k_min = 2
        k_max = self.slo_base_k
        # Clamp frac to [0,1] — elapsed_ms can be negative if arrival_time is
        # from time.time() while time.monotonic() is used here.
        frac = max(0.0, min(elapsed_ms / self.slo_latency_ms, 1.0))
        k = k_max - (k_max - k_min) * frac
        return max(k_min, round(k))

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

    # Confidence exit tracking
    conf_exit_enabled: bool = False
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
    conf_should_stop: bool = False
    conf_history: list = field(default_factory=list)  # [(total_tokens, avg_confidence)]

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

        # Extract confidence exit parameters from extra_args
        extra_args = sampling_params.extra_args or {}
        conf_exit_enabled = extra_args.get("enable_conf_exit", False)
        conf_method = extra_args.get("conf_method", "window")
        conf_window_size = extra_args.get("window_size", 2048)
        conf_ema_alpha = extra_args.get("conf_ema_alpha", 0.01)
        conf_min_tokens = extra_args.get("min_tokens", 50)
        conf_threshold = extra_args.get("conf_threshold", 12.0)
        conf_topk = extra_args.get("conf_topk", 10)
        conf_rise_ratio = extra_args.get("conf_rise_ratio", 1.5)

        # Validate that logprobs >= conf_topk when confidence exit is enabled
        if conf_exit_enabled:
            if num_logprobs is None or num_logprobs < conf_topk:
                raise ValueError(
                    f"Confidence-based early exit requires logprobs >= conf_topk. "
                    f"Got logprobs={num_logprobs}, conf_topk={conf_topk}. "
                    f"Set logprobs={conf_topk} or higher in SamplingParams."
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
            conf_exit_enabled=conf_exit_enabled,
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
            conf_should_stop=False,
            conf_history=[],
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

            # Update confidence tracking if enabled
            if self.conf_exit_enabled:
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

    def check_conf_stop(self) -> bool:
        """Check if confidence-based early exit should trigger.

        Exit if EITHER condition is met (plus directional check):
        1. Rolling window average exceeds the fixed threshold, OR
        2. Rolling window average / cumulative average >= rise_ratio
           (adaptive: recent confidence has risen significantly
           above this request's own baseline)
        Both paths require confidence trending upward (directional).

        Returns:
            True if the request should stop due to high confidence,
            False otherwise.
        """
        # Not enabled or already triggered
        if not self.conf_exit_enabled or self.conf_should_stop:
            return False

        # Not enough tokens generated yet
        if self.conf_total_tokens < self.conf_min_tokens:
            return False

        # Get current confidence estimate based on method
        if self.conf_method == "ema":
            current_conf = self.conf_ema_value
        else:
            current_conf = sum(self.conf_samples) / len(self.conf_samples)

        # Check either exit condition
        threshold_met = current_conf > self.conf_threshold

        cumulative_avg = self.conf_cumulative_sum / self.conf_cumulative_count
        rise_met = (cumulative_avg > 0
                    and current_conf / cumulative_avg >= self.conf_rise_ratio)

        if not (threshold_met or rise_met):
            return False

        # Directional check (only for window mode where we have samples)
        if self.conf_method != "ema" and self.conf_samples[-1] < self.conf_samples[0]:
            return False

        self.conf_should_stop = True
        return True

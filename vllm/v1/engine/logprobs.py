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
    conf_sample_interval: int = 24
    conf_window_size: int = 5
    conf_min_tokens: int = 50
    conf_min_samples: int = 3
    conf_threshold: float = 12.0
    conf_topk: int = 10
    conf_samples: deque = field(default_factory=deque)
    conf_total_tokens: int = 0
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
        conf_sample_interval = extra_args.get("sample_interval", 24)
        conf_window_size = extra_args.get("window_size", 5)
        conf_min_tokens = extra_args.get("min_tokens", 50)
        conf_min_samples = extra_args.get("min_samples", 3)
        conf_threshold = extra_args.get("conf_threshold", 12.0)
        conf_topk = extra_args.get("conf_topk", 10)

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
            conf_sample_interval=conf_sample_interval,
            conf_window_size=conf_window_size,
            conf_min_tokens=conf_min_tokens,
            conf_min_samples=conf_min_samples,
            conf_threshold=conf_threshold,
            conf_topk=conf_topk,
            conf_samples=deque(),
            conf_total_tokens=0,
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

        Args:
            logprobs: List of top-k logprobs for this token position,
                     sorted by rank (sampled token first at index 0).
        """
        self.conf_total_tokens += 1

        # Only compute at the sampling interval
        if self.conf_total_tokens % self.conf_sample_interval != 0:
            return

        # Compute confidence as negative mean of top-k logprobs
        # Higher confidence = more negative logprobs = model is more certain
        topk_logprobs = logprobs[: self.conf_topk]
        confidence = -sum(topk_logprobs) / len(topk_logprobs)

        # Append to rolling window
        self.conf_samples.append(confidence)

        # Pop oldest if window is full
        if len(self.conf_samples) > self.conf_window_size:
            self.conf_samples.popleft()

        # Record history for offline analysis
        if len(self.conf_samples) >= self.conf_min_samples:
            avg = sum(self.conf_samples) / len(self.conf_samples)
            self.conf_history.append((self.conf_total_tokens, avg))

    def check_conf_stop(self) -> bool:
        """Check if confidence-based early exit should trigger.

        Requires both:
        1. Rolling average confidence exceeds the threshold
        2. Confidence is trending upward (not a random spike) —
           the most recent sample must be >= the oldest sample in the window

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

        # Not enough samples collected yet
        if len(self.conf_samples) < self.conf_min_samples:
            return False

        # Compute rolling average
        avg_confidence = sum(self.conf_samples) / len(self.conf_samples)

        # Check threshold + directional: confidence must be above threshold
        # AND trending upward (latest >= oldest in window) to avoid
        # triggering on transient spikes that are already declining
        if avg_confidence > self.conf_threshold:
            if self.conf_samples[-1] >= self.conf_samples[0]:
                self.conf_should_stop = True
                return True

        return False

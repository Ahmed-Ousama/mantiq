from typing import Iterator, List, Optional


class Conversation:
    """
    A conversation over a knowledge cache: each question sees the earlier questions and answers.

    The knowledge cache is shared, not copied: after every answer it is cropped back to the
    knowledge, and the conversation keeps only its own part of the cache, which it puts back for
    the next question. So the next question reads only what is new, and many conversations can use
    one cache, one question at a time. Start one with NaiveCAG.chat().
    """

    def __init__(self, cag, past_key_values, history: Optional[list] = None, max_turns: Optional[int] = None):
        self.cag = cag
        self.past_key_values = past_key_values
        # Earlier turns, oldest first, as {"question": ..., "answer": ...}: plain data to save and resume.
        self.history: List[dict] = [_as_turn(turn) for turn in history or []]
        self.max_turns = max_turns
        self._ids: List[int] = []  # tokens after the knowledge whose keys and values are in _kept
        self._kept: Optional[list] = None  # per layer: (keys, values) of those tokens
        self._knowledge: Optional[tuple] = None  # which knowledge they were computed on

    def ask(self, question: str, max_new_tokens: int = 100, temperature: float = 0.0, top_p: float = 1.0) -> str:
        """Answer the question in the light of the conversation so far, and add the turn to it."""
        token_ids = list(self._turn(question, max_new_tokens, temperature, top_p))
        return self.cag.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def stream(self, question: str, max_new_tokens: int = 100, temperature: float = 0.0, top_p: float = 1.0) -> Iterator[str]:
        """Like ask, but yields the answer in pieces. Finish or close the stream before the next question."""
        yield from self.cag._stream_text(self._turn(question, max_new_tokens, temperature, top_p))

    def reset(self):
        """Forget the conversation; the knowledge stays."""
        self.history = []
        self._ids, self._kept, self._knowledge = [], None, None

    def _turn(self, question: str, max_new_tokens: int, temperature: float, top_p: float) -> Iterator[int]:
        cag, cache = self.cag, self.past_key_values
        origin = cache.get_seq_length()
        turns = [(turn["question"], turn["answer"]) for turn in self.history]
        if self.max_turns is not None:
            turns = turns[len(turns) - self.max_turns:] if self.max_turns > 0 else []
        ids = self._fitting_ids(turns, question, origin, max_new_tokens)

        # Put back the keys and values of the part of the conversation that is unchanged.
        reuse = 0
        knowledge = (origin, hash(tuple(getattr(cache, "token_ids", None) or ())))
        if self._kept is not None and self._knowledge == knowledge and not _has_sliding_layers(cache):
            while reuse < min(len(ids), len(self._ids)) and ids[reuse] == self._ids[reuse]:
                reuse += 1
            reuse = min(reuse, len(ids) - 1)  # the last token is fed to get the next one
            if reuse > 0:
                for layer_idx, (keys, values) in enumerate(self._kept):
                    cache.update(keys[..., :reuse, :], values[..., :reuse, :], layer_idx)

        generated: List[int] = []

        def keep(full_cache):
            # The conversation's part of the cache: what was fed (the last generated token never is).
            fed = (ids + generated)[:full_cache.get_seq_length() - origin]
            self._kept = [(layer.keys[..., origin:, :].clone(), layer.values[..., origin:, :].clone()) for layer in full_cache.layers]
            self._ids, self._knowledge = fed, knowledge

        tokens = cag._generate_from_ids(ids[reuse:], cache, max_new_tokens, temperature, top_p, before_crop=keep)
        finished = False
        try:
            for token_id in tokens:
                generated.append(token_id)
                yield token_id
            finished = True
        except GeneratorExit:
            finished = True  # the reader stopped early: keep the answer as far as they read it
            raise
        finally:
            tokens.close()
            _crop(cache, origin)
            if finished:
                answer = cag.tokenizer.decode(generated, skip_special_tokens=True).strip()
                self.history.append({"question": question.strip(), "answer": answer})

    def _fitting_ids(self, turns: list, question: str, origin: int, max_new_tokens: int) -> List[int]:
        """The conversation's tokens, dropping the oldest turns until they fit the context window."""
        limit = getattr(self.cag.model.config, "max_position_embeddings", None)
        while True:
            ids = self.cag.tokenizer.encode(self.cag._conversation_text(turns, question), add_special_tokens=False)
            if not limit or origin + len(ids) + max_new_tokens <= limit:
                return ids
            if not turns:
                raise ValueError(
                    f"The question doesn't fit: the knowledge takes {origin} of {limit} tokens, the question "
                    f"{len(ids)} and the answer up to {max_new_tokens}."
                )
            turns = turns[1:]


def _as_turn(turn) -> dict:
    if isinstance(turn, dict):
        return {"question": turn["question"], "answer": turn["answer"]}
    question, answer = turn
    return {"question": question, "answer": answer}


def _has_sliding_layers(cache) -> bool:
    # A sliding-window layer keeps only its last tokens, so it can't hand back the conversation's part.
    return any(getattr(layer, "is_sliding", False) for layer in cache.layers)


def _crop(cache, length: int):
    tokens_to_remove = cache.get_seq_length() - length
    if tokens_to_remove > 0:
        cache.crop(-tokens_to_remove)

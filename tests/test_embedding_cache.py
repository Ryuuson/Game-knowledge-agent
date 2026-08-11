import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import Agent


def test_parallel_retrievals_create_one_bge_embedder(monkeypatch):
    class FakeSentenceTransformer:
        instances = 0
        counter_lock = threading.Lock()

        def __init__(self, *args, **kwargs):
            with self.counter_lock:
                type(self).instances += 1
            time.sleep(0.05)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setattr(Agent, "RAG_BACKEND", "bge")
    monkeypatch.setattr(Agent, "_cached_embedder", None)

    with ThreadPoolExecutor(max_workers=4) as executor:
        embedders = list(executor.map(lambda _: Agent._get_embedder(), range(4)))

    assert FakeSentenceTransformer.instances == 1
    assert len({id(embedder) for embedder in embedders}) == 1

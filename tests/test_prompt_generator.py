"""Stage-2 prompt generation.

The OpenAI client is stubbed: these tests cover what we do with a response,
not what the provider returns. Response shapes are drawn from what models
actually emit when asked for JSON — the parser has to tolerate the drift.
"""

from __future__ import annotations

import json

import pytest

from wigin_tllm.scoring.prompt_generator import (
    CATEGORIES,
    SYSTEM_PROMPT,
    OpenAIPromptGenerator,
    StaticPromptGenerator,
    _parse_prompts,
)
from wigin_tllm.types import CompletionPrompt


# ─── response parsing ────────────────────────────────────────────────────


def parse(payload) -> list[CompletionPrompt]:
    return _parse_prompts(json.dumps(payload), "world_knowledge")


def test_parses_the_documented_shape():
    prompts = parse({"prompts": [{"prompt": "The sun is"}, {"prompt": "Water boils at"}]})
    assert [p.prompt for p in prompts] == ["The sun is", "Water boils at"]
    assert all(p.category == "world_knowledge" for p in prompts)


def test_parses_a_bare_list():
    assert [p.prompt for p in parse([{"prompt": "The sun is"}])] == ["The sun is"]


def test_parses_plain_strings():
    assert [p.prompt for p in parse({"prompts": ["The sun is"]})] == ["The sun is"]


def test_parses_the_completions_key():
    assert [p.prompt for p in parse({"completions": [{"prompt": "x"}]})] == ["x"]


def test_blank_and_missing_prompts_are_dropped():
    prompts = parse({"prompts": [{"prompt": "  "}, {"prompt": ""}, {}, {"prompt": "kept"}]})
    assert [p.prompt for p in prompts] == ["kept"]


def test_surrounding_whitespace_is_trimmed():
    assert parse({"prompts": [{"prompt": "  padded  "}]})[0].prompt == "padded"


def test_unknown_shape_yields_nothing():
    assert parse({"result": "no prompts here"}) == []


def test_malformed_json_raises():
    with pytest.raises(json.JSONDecodeError):
        _parse_prompts("not json", "world_knowledge")


# ─── generation ──────────────────────────────────────────────────────────


class FakeCompletions:
    def __init__(self, responses=None, fail_for=()):
        self.calls: list[dict] = []
        self.responses = responses or {}
        self.fail_for = set(fail_for)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        category = kwargs["messages"][1]["content"].split("\n", 1)[0].removeprefix("Category: ")
        if category in self.fail_for:
            raise RuntimeError("provider is unhappy")
        payload = self.responses.get(
            category, {"prompts": [{"prompt": f"{category} prompt {i}"} for i in range(2)]}
        )
        content = json.dumps(payload)
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": content})})]})


def generator(**kwargs) -> tuple[OpenAIPromptGenerator, FakeCompletions]:
    """Build a generator with a stubbed client, bypassing the real constructor."""
    fake = FakeCompletions(**{k: kwargs.pop(k) for k in ("responses", "fail_for") if k in kwargs})
    gen = OpenAIPromptGenerator.__new__(OpenAIPromptGenerator)
    gen.model = "test-model"
    gen.per_category = kwargs.get("per_category", 2)
    gen.seed = kwargs.get("seed", 1000)
    gen.categories = kwargs.get("categories", CATEGORIES)
    gen.temperature = 1.0
    gen._client = type("Client", (), {"chat": type("Chat", (), {"completions": fake})})
    return gen, fake


def test_every_category_is_requested():
    gen, fake = generator()
    gen.generate(round_id=7)
    assert len(fake.calls) == len(CATEGORIES)


def test_prompts_are_tagged_with_their_category():
    gen, _ = generator(categories={"world_knowledge": "desc", "causal_reasoning": "desc"})
    categories = {p.category for p in gen.generate(round_id=1)}
    assert categories == {"world_knowledge", "causal_reasoning"}


def test_the_system_prompt_asks_for_incomplete_text():
    gen, fake = generator(categories={"world_knowledge": "desc"})
    gen.generate(round_id=1)
    system = fake.calls[0]["messages"][0]["content"]
    assert system == SYSTEM_PROMPT
    assert "INCOMPLETE" in system
    assert "NOT questions" in system
    assert "TIMELESS" in system  # a dated prompt would be unanswerable pre-cutoff


def test_the_requested_count_and_round_reach_the_provider():
    gen, fake = generator(per_category=13, categories={"world_knowledge": "desc"})
    gen.generate(round_id=42)
    user = fake.calls[0]["messages"][1]["content"]
    assert "exactly 13 prompts" in user
    assert "Round: 42" in user


def test_each_category_gets_its_own_seed():
    """One seed for the whole round would make every category identical."""
    gen, fake = generator(seed=500)
    gen.generate(round_id=1)
    seeds = [call["seed"] for call in fake.calls]
    assert len(set(seeds)) == len(CATEGORIES)


def test_an_explicit_seed_makes_the_round_reproducible():
    first, fake_a = generator(seed=99)
    second, fake_b = generator(seed=99)
    first.generate(round_id=1)
    second.generate(round_id=1)
    assert [c["seed"] for c in fake_a.calls] == [c["seed"] for c in fake_b.calls]


def test_without_a_seed_rounds_differ():
    gen_a, fake_a = generator(seed=None)
    gen_b, fake_b = generator(seed=None)
    gen_a.generate(round_id=1)
    gen_b.generate(round_id=1)
    assert [c["seed"] for c in fake_a.calls] != [c["seed"] for c in fake_b.calls]


def test_sampling_stays_loose_for_variety():
    gen, fake = generator(categories={"world_knowledge": "desc"})
    gen.generate(round_id=1)
    assert fake.calls[0]["temperature"] == 1.0


def test_one_failing_category_does_not_lose_the_others():
    gen, _ = generator(fail_for={"world_knowledge"})
    prompts = gen.generate(round_id=1)
    assert prompts  # the other seven categories still produced prompts
    assert "world_knowledge" not in {p.category for p in prompts}


def test_every_category_failing_yields_nothing():
    gen, _ = generator(fail_for=set(CATEGORIES))
    assert gen.generate(round_id=1) == []


# ─── static generator ────────────────────────────────────────────────────


def test_static_generator_returns_its_set():
    prompts = [CompletionPrompt(prompt="a"), CompletionPrompt(prompt="b")]
    assert StaticPromptGenerator(prompts).generate(round_id=3) == prompts


def test_static_generator_hands_out_copies():
    prompts = [CompletionPrompt(prompt="a")]
    gen = StaticPromptGenerator(prompts)
    gen.generate(1).append(CompletionPrompt(prompt="b"))
    assert len(gen.generate(1)) == 1

"""Stage-2 prompt generation.

Stage 2 asks every qualified model to continue the same set of incomplete
texts. Those texts are generated fresh for each round rather than kept in a
fixed file: a static set would be learnable, and a model tuned to it would
score well without being better.

The categories are modelled on the DCLM benchmark families (HellaSwag, ARC,
PIQA and friends) so the set spans comprehension, world knowledge, reasoning
and narrative rather than testing one skill repeatedly.

Prompts are deliberately *timeless* — no dates, no recent events. A prompt
about something that happened in 2019 would be unanswerable for a model with
a 2015 cutoff, and stage 2 is measuring quality, not chronology.
"""

from __future__ import annotations

import json
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from ..types import CompletionPrompt

logger = logging.getLogger(__name__)

CATEGORIES: dict[str, str] = {
    "reading_comprehension": (
        "Generate text completion prompts that test reading comprehension. "
        "Provide a short paragraph (2-4 sentences) describing a situation, then end with an incomplete sentence "
        "that requires understanding the paragraph to complete correctly. "
        "Example: 'The school garden had tomatoes, carrots, and sunflowers. Every Friday, students took turns "
        "watering the plants and pulling weeds. By the end of summer, the tomatoes were ripe and ready to pick. "
        "The students cared for the garden by'"
    ),
    "language_understanding": (
        "Generate text completion prompts that test language understanding. "
        "Write a vivid scene or set of instructions, then stop mid-sentence at a natural point. "
        "The model must understand context, tone, and intent to continue appropriately. "
        "Example: 'The cabin stood alone at the edge of the frozen lake, its windows glowing faintly "
        "through the snow. Inside, Mara hung her coat by the door, stamped the ice from her boots, and'"
    ),
    "world_knowledge": (
        "Generate text completion prompts that test factual world knowledge. "
        "Write prompts about science, geography, history, or general facts that require knowledge to complete. "
        "Use timeless facts, not recent events. "
        "Example: 'The process by which plants use sunlight to convert carbon dioxide and water into sugar is called'"
    ),
    "commonsense_reasoning": (
        "Generate text completion prompts that test commonsense reasoning. "
        "Describe an everyday situation and end with an incomplete sentence where common sense dictates the answer. "
        "Example: 'Jamal poured orange juice into a glass until it reached the top. When he tried to add more, the juice'"
    ),
    "language_modeling": (
        "Generate text completion prompts that test narrative language modeling. "
        "Write the beginning of a story or scene with rich detail, then stop mid-sentence. "
        "The model must continue in a coherent, natural way. "
        "Example: 'The old lighthouse keeper climbed the spiral stairs each evening, his lantern swaying with each step. "
        "At the top, he'"
    ),
    "causal_reasoning": (
        "Generate text completion prompts that test causal reasoning. "
        "Describe a cause and end with an incomplete sentence about its effect, or vice versa. "
        "Example: 'The driver forgot to turn off the headlights overnight, so in the morning the car battery'"
    ),
    "logical_inference": (
        "Generate text completion prompts that test logical inference. "
        "Present premises that lead to a conclusion the model must complete. "
        "Example: 'Every mammal breathes air. Whales are mammals. Therefore, whales'"
    ),
    "temporal_reasoning": (
        "Generate text completion prompts that test understanding of time and sequence. "
        "Describe events in order and ask the model to complete what happens next or what came before. "
        "Example: 'First she mixed the flour and eggs, then she added the sugar. After stirring for two minutes, she'"
    ),
}

SYSTEM_PROMPT = """You generate text completion prompts for evaluating language models.

Rules:
- Each prompt is an INCOMPLETE sentence or paragraph that the model must continue
- Write prompts as incomplete text, NOT questions
- The prompt must stop at a natural point where a good model would produce a meaningful continuation
- Vary length: some short (10-15 words), some longer (20-50 words)
- Prompts must be TIMELESS — use general knowledge, everyday scenarios, science facts, common sense
- Do NOT reference specific dates, recent events, or anything tied to a particular year
- Make prompts diverse — different topics, scenarios, styles
- Write engaging, varied prompts — avoid generic patterns like "X wanted to do Y" or "X and his friends did Y"
- Do NOT include the expected answer
- Make prompts CHALLENGING — they should require real knowledge, reasoning, or strong language skills to complete well. Avoid trivial completions that any model could produce

Return a JSON object with a "prompts" key containing an array of objects, each with a "prompt" field."""


class PromptGenerator(ABC):
    """Produces the completion prompts for one round of stage 2."""

    @abstractmethod
    def generate(self, round_id: int) -> list[CompletionPrompt]:
        ...


class OpenAIPromptGenerator(PromptGenerator):
    """Generates a fresh prompt set per round with an OpenAI model.

    One request per category, at `temperature=1.0`: the point is variety, so
    the sampling is deliberately loose. Pass `seed` to make a round
    reproducible; leave it unset and each round draws its own.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        per_category: int = 13,
        seed: Optional[int] = None,
        categories: Optional[dict[str, str]] = None,
        temperature: float = 1.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIPromptGenerator needs the openai package: "
                "pip install 'wigin-tllm[openai]'"
            ) from e

        self.model = model or os.environ.get("PROMPT_MODEL", "gpt-4o-mini")
        self.per_category = per_category
        self.seed = seed
        self.categories = categories or CATEGORIES
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), max_retries=3)

    def generate(self, round_id: int) -> list[CompletionPrompt]:
        # A per-round seed keeps one round's set internally consistent while
        # still differing between rounds.
        base_seed = self.seed if self.seed is not None else random.getrandbits(31)

        prompts: list[CompletionPrompt] = []
        for index, (category, description) in enumerate(self.categories.items()):
            logger.info(f"Generating {self.per_category} prompts for {category}...")
            try:
                prompts.extend(
                    self._generate_category(category, description, round_id, base_seed + index)
                )
            except Exception as e:
                # One category failing should not lose the other seven.
                logger.error(f"Prompt generation failed for {category}: {type(e).__name__}: {e}")

        logger.info(f"Generated {len(prompts)} prompts for round {round_id}")
        return prompts

    def _generate_category(
        self, category: str, description: str, round_id: int, seed: int
    ) -> list[CompletionPrompt]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Category: {category}\n\n"
                        f"{description}\n\n"
                        f"Generate exactly {self.per_category} prompts. "
                        f"Round: {round_id}, seed: {seed}."
                    ),
                },
            ],
            temperature=self.temperature,
            seed=seed,
            response_format={"type": "json_object"},
        )
        return _parse_prompts(response.choices[0].message.content, category)


def _parse_prompts(content: str, category: str) -> list[CompletionPrompt]:
    """Pull prompts out of a model response, tolerating shape drift.

    The model is asked for `{"prompts": [{"prompt": ...}]}` but does not
    always oblige, so a bare list and plain strings are accepted too.
    """
    data = json.loads(content)
    if isinstance(data, list):
        items = data
    else:
        items = data.get("prompts") or data.get("completions") or []

    prompts = []
    for item in items:
        text = item if isinstance(item, str) else (item or {}).get("prompt", "")
        if isinstance(text, str) and text.strip():
            prompts.append(CompletionPrompt(prompt=text.strip(), category=category))
    return prompts


class StaticPromptGenerator(PromptGenerator):
    """Serves a fixed set. For tests, and for replaying a past round."""

    def __init__(self, prompts: Sequence[CompletionPrompt]):
        self.prompts = list(prompts)

    def generate(self, round_id: int) -> list[CompletionPrompt]:
        return list(self.prompts)

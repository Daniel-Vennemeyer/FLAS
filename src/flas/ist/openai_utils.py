"""Shared OpenAI plumbing for IST data construction and judging.

Mirrors scripts/judge_openai.py conventions (AsyncOpenAI + exponential
backoff) with two additions used by the IST pipeline: JSON-mode calls for
structured generation, and a 0-10 concept-strength judge (finer-grained than
AxBench's 0-2 concept score, needed to verify graded ladders and to measure
causal movement between responses).
"""

import asyncio
import json
import random

from openai import AsyncOpenAI, APIStatusError, APITimeoutError, RateLimitError


async def call_with_backoff(client, model, prompt, *, temperature=0.7,
                            max_tokens=2048, json_mode=False, max_retries=8):
    """Chat call with exponential backoff. Returns message content or None."""
    for attempt in range(max_retries):
        try:
            kwargs = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model, temperature=temperature, max_tokens=max_tokens,
                **kwargs)
            return resp.choices[0].message.content
        except (RateLimitError, APITimeoutError, APIStatusError):
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(min(2 ** attempt + random.random(), 60))
    return None


def parse_json_response(text):
    """Parse a JSON object from a model response, tolerating code fences."""
    if text is None:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


STRENGTH_JUDGE_PROMPT = """[System]
Please act as an impartial judge and rate how strongly the response below \
expresses the specified concept. Consider both how pervasively the concept \
appears throughout the response and how intensely it is expressed where it \
appears. Ignore fluency and whether the response answers the instruction.

Begin with a one-sentence justification. Then rate the concept strength on a \
scale from 0 to 10, where 0 means the concept is entirely absent, 1-3 means \
it appears mildly or occasionally, 4-6 means it is clearly present in much of \
the response, 7-9 means it is strongly and pervasively expressed, and 10 \
means the response is saturated with the concept. Provide your rating using \
this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Response Start]
{response}
[Response End]
"""


def extract_strength_rating(completion):
    """Parse 'Rating: [[score]]' with a 0-10 range. None on failure."""
    if completion is None or "Rating:" not in completion:
        return None
    text = completion.split("Rating:")[-1].strip().split("\n")[0].strip()
    text = (text.replace("[", "").replace("]", "").rstrip(".")
            .strip('"').strip("'").strip("*").strip())
    try:
        r = float(text)
        if 0.0 <= r <= 10.0:
            return r
    except ValueError:
        pass
    return None


async def judge_strength(client, model, concept, response):
    """Rate concept strength of a response, 0-10 (None on failure)."""
    completion = await call_with_backoff(
        client, model,
        STRENGTH_JUDGE_PROMPT.format(concept=concept, response=response),
        temperature=0.0, max_tokens=256)
    return extract_strength_rating(completion)


def make_client(api_key):
    return AsyncOpenAI(api_key=api_key)

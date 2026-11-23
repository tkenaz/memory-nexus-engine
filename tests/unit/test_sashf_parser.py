"""SASH-F parser: src/core/sashf_parser.py.

The parser runs Natasha (Russian NER/morphology) whose models ship inside the
natasha package, so nothing is downloaded; constructing it takes well under a
second. It is Russian-only by design: English text yields an empty structure.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.sashf_parser import SASHFParser
from src.models.sashf import Affect

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def parser():
    return SASHFParser()


def test_parse_returns_all_components(parser):
    result = parser.parse("Привет")
    assert set(result) == {"actors", "act", "manner", "affect", "qualia", "tags"}


def test_russian_sentence_is_fully_structured(parser):
    result = parser.parse("Вчера Анна спокойно гуляла по парку, было тепло.")
    assert "анна" in result["actors"]
    assert result["act"] == "гулять"            # lemma of the first verb
    assert result["manner"] == "спокойно"
    assert isinstance(result["affect"], Affect)
    assert result["affect"].arousal < 0          # 'спокоен' stem: low arousal
    assert "tactile" in result["qualia"]         # 'тепл'
    assert "reflection" in result["tags"]        # 'вчера' / 'был'


def test_first_and_second_person_pronouns_map_to_generic_actors(parser):
    actors = parser.parse("Я написала тебе длинное письмо.")["actors"]
    assert "assistant" in actors and "user" in actors


def test_english_text_yields_an_empty_structure(parser):
    assert parser.parse("Yesterday Alice walked slowly through Lisbon and felt joy.") == {
        "actors": [], "act": None, "manner": None, "affect": None, "qualia": [], "tags": [],
    }


def test_negative_emotion_lowers_valence(parser):
    affect = parser.parse("Сегодня была грусть и тоска.")["affect"]
    assert affect.valence < 0
    assert affect.bucket == f"V{round(affect.valence, 1):+.1f}A{round(affect.arousal, 1):+.1f}D{round(affect.dominance, 1):+.1f}"


def test_neutral_text_has_no_affect(parser):
    assert parser.parse("Купили хлеб и молоко.")["affect"] is None


@pytest.mark.parametrize(
    "text, tag",
    [
        ("Переписали код и архитектуру системы.", "technical"),
        ("Завтра план: собираюсь в горы.", "planning"),
        ("Помню, как раньше было.", "reflection"),
        ("Большая задача по проекту на работе.", "work"),
        ("Прошёл час, потом ещё неделя.", "temporal"),
    ],
)
def test_auto_tags(parser, text, tag):
    assert tag in parser.parse(text)["tags"]


@pytest.mark.parametrize(
    "text, sense",
    [
        ("Яркий свет в окне.", "visual"),
        ("Громкий звук за стеной.", "auditory"),
        ("Запах кофе на кухне.", "olfactory"),
        ("Сладкий вкус малины.", "gustatory"),
    ],
)
def test_qualia(parser, text, sense):
    assert sense in parser.parse(text)["qualia"]


_MANNER_PROBE = (
    "import sys, types\n"
    "sys.dont_write_bytecode = True\n"
    "from src.core.sashf_parser import SASHFParser\n"
    "p = SASHFParser.__new__(SASHFParser)\n"
    "p._init_manner_lexicon()\n"
    "print(p._extract_manner(types.SimpleNamespace(text='Сначала медленно, потом быстро', tokens=[])))\n"
)


def _manner_with_hash_seed(seed: int) -> str:
    out = subprocess.run(
        [sys.executable, "-c", _MANNER_PROBE],
        cwd=REPO_ROOT,
        env={"PYTHONHASHSEED": str(seed), "PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=60, check=True,
    )
    return out.stdout.strip()


def test_manner_does_not_depend_on_hash_seed():
    assert _manner_with_hash_seed(0) == _manner_with_hash_seed(3)

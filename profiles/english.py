"""English subject strategy: language learning, grammar and exams."""

from .base import SearchGuidance, SubjectProfile, SubjectStrategy


class EnglishStrategy(SubjectStrategy):
    """Plans queries and answers the way a language teacher would."""

    #: Later: ("dictionary", "conjugator") once subject tools exist.
    tools: tuple[str, ...] = ()

    profile = SubjectProfile(
        name="english",
        description=(
            "the English language: grammar, vocabulary, pronunciation, "
            "teaching and exams"
        ),
        preferred_source_types=["educational", "university", "wikipedia"],
        important_features=[
            "concrete example sentences for every rule",
            "the level the answer targets (A1-C2, school year, exam)",
            "the rule, its common exceptions and typical learner mistakes",
        ],
        verification_rules=[
            "check spellings and word meanings against a learner dictionary "
            "(dictionary.cambridge.org, oxfordlearnersdictionaries.com)",
            "prefer dated curriculum or exam references over forum answers",
            "a CEFR level claim (A1, B2) should come from a body that publishes "
            "levels: British Council, Cambridge or the Council of Europe",
        ],
        search_guidance=SearchGuidance(
            query_hints=[
                'name the grammar point exactly ("present perfect", not "a tense")',
                "add the audience: student, teacher, A2, DELF, TOEFL, IELTS",
                "add exercise-style words: examples, exercises, worksheet, explained",
                "for a learner's question, keep the level word in the query "
                '(A1, A2, B1, beginner, intermediate): the good sites are '
                "organised by level, so dropping it loses the whole result",
            ],
            lexicon=[
                "grammar",
                "tense",
                "vocabulary",
                "pronunciation",
                "CEFR",
                "ESL",
            ],
            preferred_domains=[
                "cambridge.org",
                "britannica.com",
                "khanacademy.org",
                "lelivrescolaire.fr",
                # Learner-facing sites, all confirmed reachable. Naming the
                # level in the question ("A1", "B2", "beginner") is what makes
                # these worth searching: they are organised by level, so
                # learnenglish.britishcouncil.org answers "A1 vocabulary" while
                # a general dictionary page does not.
                "learnenglish.britishcouncil.org",
                "oxfordlearnersdictionaries.com",
                "dictionary.cambridge.org",
                "bbc.co.uk",
                "voanews.com",
                "esl-lab.com",
            ],
            avoid_domains=["reddit.com", "pinterest.com", "tiktok.com"],
            require_recent=False,
        ),
    )
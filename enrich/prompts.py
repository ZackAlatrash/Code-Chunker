"""
Prompt templates for Dutch LLM enrichment of code chunks.

This module contains the system and user prompts for generating
Dutch summaries and keywords for code chunks.
"""

FILE_SYNOPSIS_SYSTEM_PROMPT = """Je bent een behulpzame code-analist. Vat kort en feitelijk samen wat het volledige bestand doet. Geen speculatie, geen ontwerpwensen. Max ~5 zinnen. Schrijf in het Nederlands."""

FILE_SYNOPSIS_USER_PROMPT = """BESTANDS PAD: {rel_path}
TAAL: {language}
SAMENVATTING DOEL: beknopt, feitelijk, zonder aannames.
BESTANDSCONTENT (samengevoegde chunks, ingekort indien nodig):
{file_text_excerpt}

Return: plain NL text synopsis (no code blocks, no markdown)."""

PER_CHUNK_SYSTEM_PROMPT = """Je schrijft korte, feitelijke samenvattingen voor codefragmenten ten behoeve van zoeken in natuurlijke taal (Nederlands).
Regels:
- 1–3 zinnen, max ~160 tekens.
- Beschrijf alleen wat het fragment daadwerkelijk doet.
- Geen speculatie of externe afhankelijkheden die niet zichtbaar zijn.
- Gebruik termen die een developer zou intypen.
- Sluit af zonder puntlijst of kopjes."""

PER_CHUNK_USER_PROMPT = """PROJECT/FILE CONTEXT (NL):
{file_synopsis_nl}

FRAGMENT METADATA:
- ast_path: {ast_path}
- node_kind: {node_kind}
- type_name: {type_name}
- function/method: {function_or_method}

CODEFRAGMENT:
{chunk_text_excerpt}

TAKEN:
1) Geef een één- of tweeregelige NL-samenvatting (max ~160 tekens).
2) Geef 5–10 NL-zoekwoorden als JSON-lijst van strings.

Formaat (JSON):
{{
  "summary_nl": "...",
  "keywords_nl": ["...", "...", "..."]
}}"""

PER_CHUNK_RETRY_PROMPT = """PROJECT/FILE CONTEXT (NL):
{file_synopsis_nl}

FRAGMENT METADATA:
- ast_path: {ast_path}
- node_kind: {node_kind}
- type_name: {type_name}
- function/method: {function_or_method}

CODEFRAGMENT:
{chunk_text_excerpt}

TAKEN:
1) Geef een één- of tweeregelige NL-samenvatting (max ~160 tekens).
2) Geef 5–10 NL-zoekwoorden als JSON-lijst van strings.

Formaat (JSON):
{{
  "summary_nl": "...",
  "keywords_nl": ["...", "...", "..."]
}}

Return JSON only."""


def get_file_synopsis_prompt(rel_path: str, language: str, file_text_excerpt: str) -> tuple[str, str]:
    """Get system and user prompts for file synopsis generation."""
    user_prompt = FILE_SYNOPSIS_USER_PROMPT.format(
        rel_path=rel_path,
        language=language,
        file_text_excerpt=file_text_excerpt
    )
    return FILE_SYNOPSIS_SYSTEM_PROMPT, user_prompt


def get_per_chunk_prompt(
    file_synopsis_nl: str,
    ast_path: str,
    node_kind: str,
    type_name: str,
    function_or_method: str,
    chunk_text_excerpt: str,
    is_retry: bool = False
) -> tuple[str, str]:
    """Get system and user prompts for per-chunk enrichment."""
    if is_retry:
        user_prompt = PER_CHUNK_RETRY_PROMPT.format(
            file_synopsis_nl=file_synopsis_nl,
            ast_path=ast_path,
            node_kind=node_kind,
            type_name=type_name,
            function_or_method=function_or_method,
            chunk_text_excerpt=chunk_text_excerpt
        )
    else:
        user_prompt = PER_CHUNK_USER_PROMPT.format(
            file_synopsis_nl=file_synopsis_nl,
            ast_path=ast_path,
            node_kind=node_kind,
            type_name=type_name,
            function_or_method=function_or_method,
            chunk_text_excerpt=chunk_text_excerpt
        )
    
    return PER_CHUNK_SYSTEM_PROMPT, user_prompt

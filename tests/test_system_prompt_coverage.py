from Agent import SYSTEM_PROMPT


def test_game_knowledge_prompt_describes_all_six_collections():
    expected_sources = (
        "game-design-wiki",
        "Game-Knowledge-Base",
        "open-game-mechanics-dataset",
        "Game_Num_Basics_And_Calc",
        "gamedev_at_home",
        "senior-game-designer",
    )

    for source in expected_sources:
        assert source in SYSTEM_PROMPT


def test_web_search_prompt_requires_verifiable_links_without_placeholder_citations():
    assert "参考链接" in SYSTEM_PROMPT
    assert "标题 — 链接" in SYSTEM_PROMPT
    assert "来源1/来源2" in SYSTEM_PROMPT

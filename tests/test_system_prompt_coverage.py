from Agent import SYSTEM_PROMPT


def test_game_knowledge_prompt_describes_distributed_collections():
    expected_sources = (
        "game-design-wiki",
        "open-game-mechanics-dataset",
        "Game_Num_Basics_And_Calc",
        "senior-game-designer",
    )

    for source in expected_sources:
        assert source in SYSTEM_PROMPT
    assert "Game-Knowledge-Base" not in SYSTEM_PROMPT
    assert "gamedev_at_home" not in SYSTEM_PROMPT

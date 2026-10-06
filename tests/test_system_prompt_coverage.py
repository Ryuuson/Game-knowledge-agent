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

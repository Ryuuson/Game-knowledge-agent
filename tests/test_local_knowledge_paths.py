from Agent import _local_knowledge_path


def test_maps_game_num_source_to_its_local_directory():
    hit = {
        "collection_label": "game_num_basics",
        "source_url": "game_num_basics/docs/Art/VFX/VFX_And_Game_Feel.md",
    }

    assert _local_knowledge_path(hit) == (
        "Game_Num_Basics_And_Calc/docs/Art/VFX/VFX_And_Game_Feel.md"
    )


def test_omits_sources_without_a_known_local_document_path():
    hit = {
        "collection_label": "game_knowledge_base",
        "source_url": "https://example.com/article",
    }

    assert _local_knowledge_path(hit) is None


def test_does_not_offer_json_mechanic_source_to_markdown_reader():
    hit = {
        "collection_label": "open_game_mechanics_dataset",
        "source_url": "open-game-mechanics-dataset/data/combat/hit_scan.json",
    }

    assert _local_knowledge_path(hit) is None

"""Low-cost, explainable domain cues for game-knowledge routing experiments."""

from dataclasses import dataclass


# A game cue wins over a non-game cue: industry terms can describe game content.
GAME_CUES = (
    "游戏",
    "玩家",
    "关卡",
    "副本",
    "boss",
    "玩法",
    "数值",
    "pve",
    "pvp",
    "roguelike",
    "模拟经营",
    "城市天际线",
)

# These are intentionally complete, high-specificity phrases rather than generic
# words such as "医院" or "建筑", which are valid subjects in game questions.
NON_GAME_CUES = (
    "医院预约挂号",
    "医院导诊",
    "在线教育",
    "教育app",
    "教育 app",
    "金融产品",
    "金融风险评级",
    "理财平台",
    "智能硬件",
    "固件升级",
    "建筑信息模型",
    "bim",
    "制造业",
    "产线排程",
    "软件开发项目",
    "商业门店",
    "app的注册流程",
)


@dataclass(frozen=True)
class DomainSignals:
    """Domain cues only; callers decide whether to clarify, search, or retrieve."""

    classification: str
    game_signals: tuple[str, ...]
    non_game_signals: tuple[str, ...]


def _matched_cues(query: str, cues: tuple[str, ...]) -> tuple[str, ...]:
    normalized = query.casefold().replace(" ", "").replace("的", "")
    return tuple(
        cue
        for cue in cues
        if cue.casefold().replace(" ", "").replace("的", "") in normalized
    )


def classify_query_domain(query: str) -> DomainSignals:
    """Return game, clear non-game, or unresolved without making a routing decision."""
    game_signals = _matched_cues(query, GAME_CUES)
    non_game_signals = _matched_cues(query, NON_GAME_CUES)
    if game_signals:
        classification = "game_context"
    elif non_game_signals:
        classification = "clear_non_game"
    else:
        classification = "unresolved"
    return DomainSignals(classification, game_signals, non_game_signals)

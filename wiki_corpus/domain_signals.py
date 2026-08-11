"""Low-cost, explainable domain cues for game-knowledge routing experiments."""

from dataclasses import dataclass


# Generic game vocabulary is often borrowed by non-game products. An explicit
# game-object marker takes precedence only when a high-specificity non-game
# context is also present.
GAME_CUES = (
    "游戏",
    "玩家",
    "关卡",
    "副本",
    "boss",
    "玩法",
    "pve",
    "pvp",
    "roguelike",
    "模拟经营",
    "城市天际线",
)

GAME_OBJECT_CUES = (
    "游戏内",
    "游戏中",
    "游戏项目",
    "游戏角色",
    "游戏里的",
)

# These are intentionally complete, high-specificity phrases rather than generic
# words such as "医院" or "建筑", which are valid subjects in game questions.
NON_GAME_CUES = (
    "医院预约挂号",
    "医院导诊",
    "在线教育",
    "线上教育",
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
    "saas",
    "短视频平台",
    "电商会员",
    "企业外包项目",
    "在线协作工具",
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
    """Return game, non-game, gamified non-game, or unresolved domain signals."""
    game_signals = _matched_cues(query, GAME_CUES)
    non_game_signals = _matched_cues(query, NON_GAME_CUES)
    game_object_signals = _matched_cues(query, GAME_OBJECT_CUES)
    if non_game_signals and game_signals and not game_object_signals:
        classification = "gamified_non_game"
    elif game_signals:
        classification = "game_context"
    elif non_game_signals:
        classification = "clear_non_game"
    else:
        classification = "unresolved"
    return DomainSignals(classification, game_signals, non_game_signals)

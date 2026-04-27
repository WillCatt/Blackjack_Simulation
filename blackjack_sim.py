"""
Blackjack Strategy Simulator
=============================
Monte Carlo simulation engine for comparing betting strategies under
configurable table rules. Outputs JSON consumed by the D3 visualisation layer.

Strategies implemented:
    - Flat Bet (control)
    - Martingale (loss-doubling)
    - Paroli / Anti-Martingale (win-doubling, 3-win cap)
    - 1-3-2-6 system (positive progression)
    - Oscar's Grind (conservative grind)
    - Hi-Lo card counting (true-count proportional sizing)
        - Hi-Lo CHEATING: same hand-play as everyone else, but the cheater always
      knows the EXACT true count of the remaining shoe (perfect recall + no
      counting errors + no deck-estimation slop), so they bet with full
      confidence on the optimal ramp.
    - MIT TEAM: full 3-role professional team (Spotter, Controller, Big
      Player) with independent bankrolls and a collated team P/L.

Author: William
"""

from __future__ import annotations

import json
import random
import sys
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
from pathlib import Path


# ─── Card & Shoe ─────────────────────────────────────────────────────────────

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["♠", "♥", "♦", "♣"]

# Hi-Lo count tags
HILO_COUNT = {
    "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}


def card_value(rank: str) -> int:
    """Return the point value of a card. Aces return 11 (adjusted later)."""
    if rank == "A":
        return 11
    if rank in ("K", "Q", "J", "10"):
        return 10
    return int(rank)


def hand_total(cards: List[str]) -> int:
    """Calculate optimal hand total, reducing aces as needed."""
    total = sum(card_value(c) for c in cards)
    aces = sum(1 for c in cards if c == "A")
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def is_soft(cards: List[str]) -> bool:
    """Return True if the hand contains a usable ace (soft hand)."""
    total = sum(card_value(c) for c in cards)
    aces = sum(1 for c in cards if c == "A")
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return aces > 0 and total <= 21


class Shoe:
    """
    Multi-deck shoe with cut-card mechanics.

    Parameters
    ----------
    num_decks : int
        Number of 52-card decks in the shoe.
    penetration : float
        Percentage of shoe dealt before reshuffling (0-100).
    """

    def __init__(self, num_decks: int = 6, penetration: float = 75.0):
        self.num_decks = num_decks
        self.penetration = penetration
        self._build_and_shuffle()

    def _build_and_shuffle(self):
        self.cards = RANKS * 4 * self.num_decks
        random.shuffle(self.cards)
        self.index = 0
        self.cut_card = int(len(self.cards) * self.penetration / 100)

    def deal(self) -> str:
        """Draw one card from the shoe."""
        card = self.cards[self.index]
        self.index += 1
        return card

    def needs_reshuffle(self) -> bool:
        """Check if we've passed the cut card."""
        return self.index >= self.cut_card

    def reshuffle(self):
        """Rebuild and shuffle the shoe."""
        self._build_and_shuffle()

    @property
    def cards_remaining(self) -> int:
        return len(self.cards) - self.index

    @property
    def decks_remaining(self) -> float:
        return max(0.5, self.cards_remaining / 52)


# ─── Basic Strategy ─────────────────────────────────────────────────────────

class Action(Enum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"


def basic_strategy(player_cards: List[str], dealer_up: str, can_double: bool = True, can_split: bool = True) -> Action:
    """
    Simplified basic strategy lookup.
    Covers hard totals, soft totals, and pair splitting.
    """
    pt = hand_total(player_cards)
    dv = card_value(dealer_up)
    soft = is_soft(player_cards)

    # ── Pair splitting ──
    if can_split and len(player_cards) == 2 and player_cards[0] == player_cards[1]:
        pv = card_value(player_cards[0])
        if pv in (11, 8):
            return Action.SPLIT
        if pv == 9 and dv in range(2, 7):
            return Action.SPLIT
        if pv == 7 and dv in range(2, 8):
            return Action.SPLIT
        if pv == 6 and dv in range(2, 7):
            return Action.SPLIT
        if pv in (2, 3) and dv in range(2, 8):
            return Action.SPLIT
        if pv == 4 and dv in (5, 6):
            return Action.SPLIT

    # ── Soft totals ──
    if soft:
        if pt >= 19:
            return Action.STAND
        if pt == 18:
            if dv >= 9:
                return Action.HIT
            if dv in range(3, 7) and can_double:
                return Action.DOUBLE
            return Action.STAND
        if pt == 17 and dv in range(3, 7) and can_double:
            return Action.DOUBLE
        if pt in (15, 16) and dv in range(4, 7) and can_double:
            return Action.DOUBLE
        if pt in (13, 14) and dv in (5, 6) and can_double:
            return Action.DOUBLE
        return Action.HIT

    # ── Hard totals ──
    if pt >= 17:
        return Action.STAND
    if 13 <= pt <= 16 and dv in range(2, 7):
        return Action.STAND
    if pt == 12 and dv in range(4, 7):
        return Action.STAND
    if pt == 11 and can_double:
        return Action.DOUBLE
    if pt == 10 and dv <= 9 and can_double:
        return Action.DOUBLE
    if pt == 9 and dv in range(3, 7) and can_double:
        return Action.DOUBLE
    return Action.HIT


# ─── Hand Resolution ────────────────────────────────────────────────────────

@dataclass
class HandResult:
    """Result of a single hand (or split sub-hand)."""
    player_cards: List[str]
    dealer_cards: List[str]
    player_total: int
    dealer_total: int
    net_units: float        # net result in bet-unit multiples
    doubled: bool = False
    is_blackjack: bool = False
    is_split: bool = False
    cards_seen: List[str] = field(default_factory=list)


def play_hand(shoe: Shoe, dealer_stands_s17: bool = True) -> HandResult:
    """
    Play one complete hand using basic strategy.
    Returns the result with all cards seen (for counting).
    """
    # Initial deal
    p = [shoe.deal(), shoe.deal()]
    d = [shoe.deal(), shoe.deal()]
    all_cards = p + d

    dealer_up = d[0]
    pt, dt = hand_total(p), hand_total(d)

    # ── Natural blackjack checks ──
    if pt == 21:
        net = 0.0 if dt == 21 else 1.5
        return HandResult(p, d, pt, dt, net, is_blackjack=(dt != 21), cards_seen=all_cards)
    if dt == 21:
        return HandResult(p, d, pt, dt, -1.0, cards_seen=all_cards)

    # ── Player action loop (simplified: handle one split level) ──
    def play_player(cards: List[str], can_split: bool) -> List[Tuple[List[str], bool]]:
        """Returns list of (final_cards, was_doubled) tuples."""
        action = basic_strategy(cards, dealer_up, can_double=(len(cards) == 2), can_split=can_split)

        if action == Action.SPLIT and can_split:
            h1 = [cards[0], shoe.deal()]
            h2 = [cards[1], shoe.deal()]
            all_cards.extend([h1[-1], h2[-1]])
            return play_player(h1, False) + play_player(h2, False)

        doubled = False
        while True:
            action = basic_strategy(cards, dealer_up, can_double=(len(cards) == 2 and not doubled), can_split=False)
            if action == Action.STAND:
                break
            if action == Action.DOUBLE:
                c = shoe.deal()
                cards.append(c)
                all_cards.append(c)
                doubled = True
                break
            # HIT
            c = shoe.deal()
            cards.append(c)
            all_cards.append(c)
            if hand_total(cards) > 21:
                break
        return [(cards, doubled)]

    player_hands = play_player(list(p), can_split=True)

    # ── Dealer plays if any player hand is alive ──
    any_alive = any(hand_total(h) <= 21 for h, _ in player_hands)
    if any_alive:
        while True:
            dt = hand_total(d)
            if dt > 17:
                break
            if dt == 17:
                if dealer_stands_s17 or not is_soft(d):
                    break
            c = shoe.deal()
            d.append(c)
            all_cards.append(c)
    dt = hand_total(d)

    # ── Resolve each player hand ──
    total_net = 0.0
    for cards, was_doubled in player_hands:
        mult = 2.0 if was_doubled else 1.0
        pt_h = hand_total(cards)
        if pt_h > 21:
            total_net -= mult
        elif dt > 21:
            total_net += mult
        elif pt_h > dt:
            total_net += mult
        elif pt_h < dt:
            total_net -= mult
        # push = 0

    return HandResult(
        player_cards=player_hands[0][0] if len(player_hands) == 1 else [c for h, _ in player_hands for c in h],
        dealer_cards=d,
        player_total=hand_total(player_hands[0][0]) if len(player_hands) == 1 else 0,
        dealer_total=dt,
        net_units=total_net,
        doubled=any(dd for _, dd in player_hands),
        is_split=len(player_hands) > 1,
        cards_seen=all_cards,
    )


# ─── Helpers for the Cheating Counter ───────────────────────────────────────

def _exact_running_count(shoe: Shoe) -> int:
    """
    Compute the Hi-Lo running count of the cards REMAINING in the shoe.
    The cheating counter has perfect recall and never miscounts, so they
    effectively know this value at every moment, even at the start of a
    fresh shoe (where it's always zero).
    """
    rc = 0
    for i in range(shoe.index, len(shoe.cards)):
        rc -= HILO_COUNT.get(shoe.cards[i], 0)
    return rc


# ─── Betting Strategies ─────────────────────────────────────────────────────

class BettingStrategy:
    """Base class for betting strategies."""
    key: str = "base"
    name: str = "Base"

    def __init__(self, table_min: float):
        self.table_min = table_min

    def get_bet(self, **kwargs) -> float:
        return self.table_min

    def update(self, won: bool, net_units: float, cards_seen: List[str] = None):
        pass

    def on_reshuffle(self):
        pass


class FlatBet(BettingStrategy):
    key = "flat"
    name = "Flat Bet"

    def get_bet(self, **kwargs):
        return self.table_min


class Martingale(BettingStrategy):
    key = "martingale"
    name = "Martingale"

    def __init__(self, table_min):
        super().__init__(table_min)
        self.consecutive_losses = 0

    def get_bet(self, **kwargs):
        return self.table_min * (2 ** self.consecutive_losses)

    def update(self, won, net_units, cards_seen=None):
        self.consecutive_losses = 0 if won else self.consecutive_losses + 1


class Paroli(BettingStrategy):
    key = "paroli"
    name = "Paroli"

    def __init__(self, table_min):
        super().__init__(table_min)
        self.consecutive_wins = 0

    def get_bet(self, **kwargs):
        return self.table_min * (2 ** min(self.consecutive_wins, 2))

    def update(self, won, net_units, cards_seen=None):
        if won:
            self.consecutive_wins = min(self.consecutive_wins + 1, 3)
            if self.consecutive_wins >= 3:
                self.consecutive_wins = 0
        else:
            self.consecutive_wins = 0


class OneTwoThreeSix(BettingStrategy):
    key = "oneThreeTwoSix"
    name = "1-3-2-6"

    SEQUENCE = [1, 3, 2, 6]

    def __init__(self, table_min):
        super().__init__(table_min)
        self.step = 0

    def get_bet(self, **kwargs):
        return self.table_min * self.SEQUENCE[self.step]

    def update(self, won, net_units, cards_seen=None):
        if won:
            self.step = min(self.step + 1, 3)
        else:
            self.step = 0


class OscarsGrind(BettingStrategy):
    key = "oscar"
    name = "Oscar's Grind"

    def __init__(self, table_min):
        super().__init__(table_min)
        self.current_bet = 1
        self.cycle_profit = 0.0

    def get_bet(self, **kwargs):
        return self.table_min * self.current_bet

    def update(self, won, net_units, cards_seen=None):
        payout = self.current_bet * (1 if won else -1)
        self.cycle_profit += payout
        if self.cycle_profit >= 1:
            # Cycle complete — reset
            self.current_bet = 1
            self.cycle_profit = 0.0
        elif won:
            self.current_bet += 1


class HiLoCount(BettingStrategy):
    key = "hilo"
    name = "Hi-Lo Count"

    def __init__(self, table_min):
        super().__init__(table_min)
        self.running_count = 0
        self._decks_remaining = 4.0

    def get_bet(self, decks_remaining: float = 4.0, **kwargs):
        self._decks_remaining = max(0.5, decks_remaining)
        tc = self.running_count / self._decks_remaining
        if tc <= 1:
            multiplier = 1
        elif tc <= 2:
            multiplier = 2
        elif tc <= 3:
            multiplier = 4
        elif tc <= 4:
            multiplier = 8
        else:
            multiplier = 12
        return self.table_min * multiplier

    def update(self, won, net_units, cards_seen=None):
        if cards_seen:
            for card in cards_seen:
                self.running_count += HILO_COUNT.get(card, 0)

    def on_reshuffle(self):
        self.running_count = 0


class HiLoCheat(BettingStrategy):
    """
    'Cheating' Hi-Lo counter.

    The ONLY thing the cheater does differently is **know the true count
    perfectly at every single moment** — they don't have to mentally tally
    cards as they fly by, they don't make estimation errors on
    decks-remaining, and they don't need a shoe to "warm up" before
    trusting the count. They play the same basic strategy as everyone
    else; the cheat is purely informational.

    Implementation: instead of accumulating a running count from the cards
    we've seen, we read it straight off the remaining shoe composition
    each turn. (In our deterministic engine this gives the exact same
    numeric value as the cards-seen tally, but conceptually it represents
    perfect recall — and it lets us bet with full confidence.)

    Because the count is rock-solid, the cheater bets on a much steeper
    ramp than a real-world counter would dare to.
    """
    key = "hilo_cheat"
    name = "Hi-Lo (Cheating)"

    def __init__(self, table_min):
        super().__init__(table_min)
        self.running_count = 0       # mirrored for live-replay display
        self._decks_remaining = 4.0
        self._shoe_ref = None        # optional handle for perfect-count lookups

    def attach_shoe(self, shoe: Shoe):
        """Called by the engine each round so we can peek at composition."""
        self._shoe_ref = shoe
        self.running_count = _exact_running_count(shoe)
        self._decks_remaining = max(0.5, shoe.decks_remaining)

    def get_bet(self, decks_remaining: float = 4.0, **kwargs):
        self._decks_remaining = max(0.5, decks_remaining)
        tc = self.running_count / self._decks_remaining
        # Aggressive ramp — the cheater knows EV is positive when TC is positive.
        if tc <= 0:
            multiplier = 1
        elif tc <= 1:
            multiplier = 4
        elif tc <= 2:
            multiplier = 12
        elif tc <= 3:
            multiplier = 25
        elif tc <= 4:
            multiplier = 40
        else:
            multiplier = 60
        return self.table_min * multiplier

    def update(self, won, net_units, cards_seen=None):
        # If we have a shoe reference we re-read the count perfectly;
        # otherwise we fall back to the standard cards-seen tally so the
        # class still behaves correctly when used outside the engine.
        if self._shoe_ref is not None:
            self.running_count = _exact_running_count(self._shoe_ref)
        elif cards_seen:
            for card in cards_seen:
                self.running_count += HILO_COUNT.get(card, 0)

    def on_reshuffle(self):
        # Fresh shoe → count truly is zero; re-sync from the shoe if attached.
        if self._shoe_ref is not None:
            self.running_count = _exact_running_count(self._shoe_ref)
        else:
            self.running_count = 0

    def true_count(self) -> float:
        return self.running_count / max(0.5, self._decks_remaining)


STRATEGY_CLASSES = {
    "flat": FlatBet,
    "martingale": Martingale,
    "paroli": Paroli,
    "oneThreeTwoSix": OneTwoThreeSix,
    "oscar": OscarsGrind,
    "hilo": HiLoCount,
    "hilo_cheat": HiLoCheat,
    "mit_team": None,   # MIT team uses a custom multi-actor runner; see run_mit_team_sim
}

STRATEGY_META = {
    "flat":            {"color": "#4fc3f7", "desc": "Bet the same amount every hand. The control group."},
    "martingale":      {"color": "#ef5350", "desc": "Double after every loss, reset after a win."},
    "paroli":          {"color": "#66bb6a", "desc": "Double after a win (3-win cap), reset on a loss."},
    "oneThreeTwoSix":  {"color": "#ffa726", "desc": "Bet 1→3→2→6 units on consecutive wins."},
    "oscar":           {"color": "#ab47bc", "desc": "Raise by 1 unit after a win; reset when cycle profits 1 unit."},
    "hilo":            {"color": "#26c6da", "desc": "Bet proportional to the Hi-Lo true count."},
    "hilo_cheat":      {"color": "#ffd54f", "desc": "Oracle player — peeks hole card, perfect index plays, ultra-aggressive ramp."},
    "mit_team":        {"color": "#e040fb", "desc": "MIT-style 3-person team: Spotter counts, Big Player swoops on hot shoes."},
}

# Friendly display names for each MIT-team role.
MIT_ROLES = ("spotter", "controller", "big_player")
MIT_ROLE_LABELS = {
    "spotter":    "Spotter",
    "controller": "Controller",
    "big_player": "Big Player",
}
MIT_ROLE_COLORS = {
    "spotter":    "#4fc3f7",   # cool blue — sits at the table grinding
    "controller": "#66bb6a",   # green — coordinates and feeds bets
    "big_player": "#e040fb",   # vivid magenta — swoops in
    "combined":   "#ffd54f",   # gold — team total
}


# ─── Simulation Engine ──────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """All configurable parameters for a simulation run."""
    num_decks: int = 6
    penetration: float = 75.0
    start_bankroll: float = 10_000
    num_hands: int = 2_000
    table_min: float = 25
    table_max: float = 5_000
    bj_pays: str = "3:2"
    dealer_stands_s17: bool = True
    num_sims: int = 50
    strategies: List[str] = field(default_factory=lambda: ["flat", "martingale", "hilo"])
    # Risk controls. None = disabled.
    # stop_loss_pct: terminate run early when bankroll falls by this % of start
    #   (e.g. 25 → stop when bankroll <= start_bankroll * 0.75)
    # take_profit_pct: terminate run early when bankroll grows by this % above start
    #   (e.g. 50 → stop when bankroll >= start_bankroll * 1.50)
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None


@dataclass
class HandSnapshot:
    """Per-hand data point for live replay mode."""
    hand_num: int
    bankroll: float
    bet: float
    net: float
    player_cards: List[str]
    dealer_cards: List[str]
    player_total: int
    dealer_total: int
    result: str  # "win", "loss", "push", "blackjack"
    running_count: Optional[int] = None
    true_count: Optional[float] = None


def run_mit_team_sim(config: "SimConfig", record_hands: bool = False) -> Dict:
    """
    Run one Monte Carlo simulation of the classic MIT-style 3-person team.

    Roles
    -----
    Spotter
        Sits at the table playing flat table-min bets, counting silently.
        Their job is to track the count, not to win — they will hover
        slightly negative on average due to house edge but they generate
        the signal.

    Controller
        Verifies the count by playing modest bets (1× → 3× table_min).
        Acts as the redundancy check on the spotter and is the conduit
        between the count and the Big Player. Mildly profitable in hot
        shoes, near-flat in cold ones.

    Big Player
        Wanders the casino looking like a tourist. Only joins when the
        team signals a hot shoe (true count ≥ +2) and then bets HUGE
        — 15× to 60× the table minimum depending on the count. They
        carry the team's actual edge.

    Per "round" (one entry in the configured `num_hands`), all three actors
    play sequentially against the same shoe. Their bankrolls update
    independently and the team total = spotter + controller + big_player.
    """
    shoe = Shoe(config.num_decks, config.penetration)
    bj_multiplier = 1.5 if config.bj_pays == "3:2" else 1.2

    # Split the starting bankroll equally between the three actors.
    role_br = {role: config.start_bankroll / 3.0 for role in MIT_ROLES}
    starting_team = sum(role_br.values())

    running_count = 0
    max_team = starting_team
    max_drawdown = 0.0
    ruined = False
    stopped_loss = False
    stopped_profit = False
    hands_played = 0

    # SL / TP thresholds for the team total
    sl_threshold = None
    if config.stop_loss_pct is not None and config.stop_loss_pct > 0:
        sl_threshold = config.start_bankroll * (1.0 - config.stop_loss_pct / 100.0)
    tp_threshold = None
    if config.take_profit_pct is not None and config.take_profit_pct > 0:
        tp_threshold = config.start_bankroll * (1.0 + config.take_profit_pct / 100.0)

    sample_interval = max(1, config.num_hands // 1000)
    history = [{
        "hand": 0,
        "bankroll": round(starting_team, 2),
        "spotter": round(role_br["spotter"], 2),
        "controller": round(role_br["controller"], 2),
        "big_player": round(role_br["big_player"], 2),
    }]
    hands: List[Dict] = []

    # Aggregate stats — Big Player hands count toward the team's "wins".
    bp_wins = bp_losses = bp_pushes = bp_blackjacks = bp_doubles = 0
    bp_hands_played = 0
    spot_wins = spot_losses = ctrl_wins = ctrl_losses = 0

    def _safe_reshuffle():
        nonlocal running_count
        shoe.reshuffle()
        running_count = 0

    def _resolve(role: str, bet: float, use_cheat: bool = False) -> Tuple[float, "HandResult"]:
        """Play one hand for `role` and return (payout, raw_result)."""
        nonlocal running_count
        if shoe.needs_reshuffle() or shoe.cards_remaining < 15:
            _safe_reshuffle()
        # BP optionally uses the cheat engine — they're the highly trained
        # one who knows index plays cold; we approximate with the peek engine.
        play_fn = play_hand_cheat if use_cheat else play_hand
        r = play_fn(shoe, config.dealer_stands_s17)
        # Update count from cards we just observed.
        for c in r.cards_seen:
            running_count += HILO_COUNT.get(c, 0)
        if r.is_blackjack and r.net_units > 0:
            payout = bet * bj_multiplier
        else:
            payout = r.net_units * bet
        role_br[role] = round(role_br[role] + payout, 2)
        return payout, r

    for h in range(config.num_hands):
        if shoe.needs_reshuffle():
            _safe_reshuffle()

        team_br = sum(role_br.values())

        # ── Ruin check (team-level) ──
        # The team is ruined when the combined bankroll cannot legally cover
        # at least the spotter sitting in for one more table-min round.
        if team_br < config.table_min or team_br <= 0:
            ruined = True
            break

        # ── Stop-loss / take-profit gates ──
        if sl_threshold is not None and team_br <= sl_threshold:
            stopped_loss = True
            break
        if tp_threshold is not None and team_br >= tp_threshold:
            stopped_profit = True
            break

        hands_played += 1
        tc_before = running_count / shoe.decks_remaining

        # ── SPOTTER plays flat table_min ──
        spot_bet = min(config.table_min, max(role_br["spotter"], 0))
        spot_payout, spot_r = _resolve("spotter", spot_bet) if spot_bet > 0 else (0.0, None)
        if spot_r is not None:
            if spot_payout > 0: spot_wins += 1
            elif spot_payout < 0: spot_losses += 1

        tc_mid = running_count / shoe.decks_remaining

        # ── CONTROLLER plays a moderate bet, scaled by count ──
        if tc_mid >= 3:
            ctrl_mult = 4
        elif tc_mid >= 2:
            ctrl_mult = 3
        elif tc_mid >= 1:
            ctrl_mult = 2
        else:
            ctrl_mult = 1
        ctrl_bet = min(config.table_min * ctrl_mult, config.table_max, max(role_br["controller"], 0))
        ctrl_payout, ctrl_r = (_resolve("controller", ctrl_bet) if ctrl_bet > 0 else (0.0, None))
        if ctrl_r is not None:
            if ctrl_payout > 0: ctrl_wins += 1
            elif ctrl_payout < 0: ctrl_losses += 1

        tc_after = running_count / shoe.decks_remaining

        # ── BIG PLAYER joins only on a hot shoe (TC ≥ +2) ──
        bp_payout = 0.0
        bp_bet = 0.0
        bp_r = None
        bp_label = "scout"   # spotter/controller round, BP not playing
        if tc_after >= 2 and role_br["big_player"] > 0:
            # Aggressive ramp keyed off true count.
            if tc_after >= 5:
                bp_mult = 60
            elif tc_after >= 4:
                bp_mult = 40
            elif tc_after >= 3:
                bp_mult = 25
            else:
                bp_mult = 15
            bp_bet = min(config.table_min * bp_mult, config.table_max, role_br["big_player"])
            bp_payout, bp_r = _resolve("big_player", bp_bet, use_cheat=False)
            bp_hands_played += 1
            if bp_r.net_units > 0:
                if bp_r.is_blackjack:
                    bp_blackjacks += 1
                    bp_label = "blackjack"
                else:
                    bp_wins += 1
                    bp_label = "win"
            elif bp_r.net_units < 0:
                bp_losses += 1
                bp_label = "loss"
            else:
                bp_pushes += 1
                bp_label = "push"
            if bp_r.doubled:
                bp_doubles += 1

        team_br = sum(role_br.values())
        max_team = max(max_team, team_br)
        if max_team > 0:
            floor_br = max(team_br, 0)
            dd = (max_team - floor_br) / max_team
            dd = max(0.0, min(1.0, dd))
            max_drawdown = max(max_drawdown, dd)

        if h % sample_interval == 0 or h == config.num_hands - 1:
            history.append({
                "hand": h + 1,
                "bankroll": round(team_br, 2),
                "spotter": round(role_br["spotter"], 2),
                "controller": round(role_br["controller"], 2),
                "big_player": round(role_br["big_player"], 2),
            })

        if record_hands:
            # Pick the most "interesting" hand to display: BP if active,
            # otherwise controller, otherwise spotter.
            display_r = bp_r or ctrl_r or spot_r
            display_bet = bp_bet if bp_r else (ctrl_bet if ctrl_r else spot_bet)
            display_actor = "big_player" if bp_r else ("controller" if ctrl_r else "spotter")
            snap = {
                "hand_num": h + 1,
                "bankroll": round(team_br, 2),
                "bet": round(display_bet, 2),
                "net": round(spot_payout + ctrl_payout + bp_payout, 2),
                "result": bp_label if bp_r else ("active" if ctrl_r or spot_r else "scout"),
                "player_cards": display_r.player_cards if display_r else [],
                "dealer_cards": display_r.dealer_cards if display_r else [],
                "player_total": display_r.player_total if display_r else 0,
                "dealer_total": display_r.dealer_total if display_r else 0,
                "running_count": running_count,
                "true_count": round(running_count / shoe.decks_remaining, 2),
                "active_actor": display_actor,
                "bp_active": bp_r is not None,
                # Per-role detail
                "spotter_br": round(role_br["spotter"], 2),
                "controller_br": round(role_br["controller"], 2),
                "big_player_br": round(role_br["big_player"], 2),
                "spotter_bet": round(spot_bet, 2),
                "controller_bet": round(ctrl_bet, 2),
                "big_player_bet": round(bp_bet, 2),
                "spotter_pl": round(spot_payout, 2),
                "controller_pl": round(ctrl_payout, 2),
                "big_player_pl": round(bp_payout, 2),
            }
            hands.append(snap)

    bp_total = bp_wins + bp_losses + bp_pushes + bp_blackjacks
    final_team = sum(role_br.values())
    if history[-1]["hand"] != hands_played:
        history.append({
            "hand": hands_played,
            "bankroll": round(final_team, 2),
            "spotter": round(role_br["spotter"], 2),
            "controller": round(role_br["controller"], 2),
            "big_player": round(role_br["big_player"], 2),
        })
    return {
        "history": history,
        "hands": hands if record_hands else [],
        "final_bankroll": round(final_team, 2),
        "team_breakdown": {
            "spotter":    round(role_br["spotter"], 2),
            "controller": round(role_br["controller"], 2),
            "big_player": round(role_br["big_player"], 2),
        },
        "max_drawdown": round(max_drawdown, 4),
        "ruined": ruined,
        "stopped_loss": stopped_loss,
        "stopped_profit": stopped_profit,
        "hands_played": hands_played,
        "stats": {
            # "Headline" stats use Big Player hands — they're the team's edge.
            "total_hands": bp_total,
            "wins": bp_wins,
            "losses": bp_losses,
            "pushes": bp_pushes,
            "blackjacks": bp_blackjacks,
            "doubles": bp_doubles,
            "win_rate": round((bp_wins + bp_blackjacks) / max(bp_total, 1), 4),
            "bp_hands_played": bp_hands_played,
            "bp_play_rate": round(bp_hands_played / max(config.num_hands, 1), 4),
            "spotter_record": {"wins": spot_wins, "losses": spot_losses},
            "controller_record": {"wins": ctrl_wins, "losses": ctrl_losses},
        },
    }


def run_single_sim(
    config: SimConfig,
    strategy_key: str,
    record_hands: bool = False,
) -> Dict:
    """
    Run one complete simulation for a single strategy.

    Parameters
    ----------
    config : SimConfig
    strategy_key : str
    record_hands : bool
        If True, record per-hand snapshots for live replay.

    Returns
    -------
    dict with keys: history (sampled bankroll), hands (if recorded),
    final_bankroll, max_drawdown, ruined, stats.
    """
    # MIT team is a multi-actor strategy and has its own runner.
    if strategy_key == "mit_team":
        return run_mit_team_sim(config, record_hands=record_hands)

    shoe = Shoe(config.num_decks, config.penetration)
    strategy = STRATEGY_CLASSES[strategy_key](config.table_min)
    bankroll = config.start_bankroll
    max_bankroll = bankroll
    max_drawdown = 0.0
    ruined = False
    stopped_loss = False
    stopped_profit = False

    # Pre-compute SL / TP thresholds (None when disabled)
    sl_threshold = None
    if config.stop_loss_pct is not None and config.stop_loss_pct > 0:
        sl_threshold = config.start_bankroll * (1.0 - config.stop_loss_pct / 100.0)
    tp_threshold = None
    if config.take_profit_pct is not None and config.take_profit_pct > 0:
        tp_threshold = config.start_bankroll * (1.0 + config.take_profit_pct / 100.0)

    bj_multiplier = 1.5 if config.bj_pays == "3:2" else 1.2
    # All strategies use the standard hand-resolution engine — the
    # "cheating" Hi-Lo only differs in how it knows the count.
    play_fn = play_hand

    # Sampling for bankroll chart (cap at ~1000 points)
    sample_interval = max(1, config.num_hands // 1000)
    history = [{"hand": 0, "bankroll": bankroll}]
    hands = []
    hands_played = 0   # track actual hands dealt (may be < num_hands if SL/TP/ruin)

    wins, losses, pushes, blackjacks, doubles = 0, 0, 0, 0, 0

    for h in range(config.num_hands):
        if shoe.needs_reshuffle():
            shoe.reshuffle()
            strategy.on_reshuffle()

        # Cheating Hi-Lo reads the count directly off the shoe before sizing.
        if isinstance(strategy, HiLoCheat):
            strategy.attach_shoe(shoe)

        # Determine the bet the strategy *wants* to place this round.
        raw_bet = strategy.get_bet(decks_remaining=shoe.decks_remaining)
        # Floor at the table min (you can't bet less than the minimum).
        next_required_bet = max(raw_bet, config.table_min)

        # ── Ruin check ──
        # A run is ruined the moment the player can no longer legally place
        # the bet their strategy demands:
        #   • bankroll has fallen below the table minimum, OR
        #   • the strategy needs more than the table maximum, OR
        #   • the strategy needs more than the player has on hand.
        if (
            bankroll < config.table_min
            or next_required_bet > config.table_max
            or bankroll < next_required_bet
        ):
            ruined = True
            break

        # ── Stop-loss / take-profit gates ──
        if sl_threshold is not None and bankroll <= sl_threshold:
            stopped_loss = True
            break
        if tp_threshold is not None and bankroll >= tp_threshold:
            stopped_profit = True
            break

        bet = next_required_bet  # all guards passed — required bet is legal

        # Safety check — ensure enough cards
        if shoe.cards_remaining < 15:
            shoe.reshuffle()
            strategy.on_reshuffle()

        # Play the hand
        result = play_fn(shoe, config.dealer_stands_s17)
        hands_played += 1

        # Calculate payout
        if result.is_blackjack and result.net_units > 0:
            payout = bet * bj_multiplier
        else:
            payout = result.net_units * bet

        bankroll = round(bankroll + payout, 2)
        max_bankroll = max(max_bankroll, bankroll)
        # Drawdown is the % decline of bankroll from its running peak,
        # clamped to [0, 1] in case of negative bankroll due to rounding.
        floor_br = max(bankroll, 0)
        dd = (max_bankroll - floor_br) / max_bankroll if max_bankroll > 0 else 0
        dd = max(0.0, min(1.0, dd))
        max_drawdown = max(max_drawdown, dd)

        # Classify result
        won = result.net_units > 0
        if result.net_units > 0:
            if result.is_blackjack:
                blackjacks += 1
                result_label = "blackjack"
            else:
                wins += 1
                result_label = "win"
        elif result.net_units < 0:
            losses += 1
            result_label = "loss"
        else:
            pushes += 1
            result_label = "push"

        if result.doubled:
            doubles += 1

        # Update strategy state
        strategy.update(won, result.net_units, result.cards_seen)

        # Record sampled history
        if h % sample_interval == 0 or h == config.num_hands - 1:
            history.append({"hand": h + 1, "bankroll": round(bankroll, 2)})

        # Record full hand for live replay
        if record_hands:
            snap = {
                "hand_num": h + 1,
                "bankroll": round(bankroll, 2),
                "bet": round(bet, 2),
                "net": round(payout, 2),
                "player_cards": result.player_cards,
                "dealer_cards": result.dealer_cards,
                "player_total": result.player_total,
                "dealer_total": result.dealer_total,
                "result": result_label,
            }
            if strategy_key in ("hilo", "hilo_cheat"):
                snap["running_count"] = strategy.running_count
                snap["true_count"] = round(strategy.running_count / shoe.decks_remaining, 2)
            hands.append(snap)

    total_hands = wins + losses + pushes + blackjacks
    # Always record the final bankroll as the last history point so curves
    # have a reliable termination value even if the loop exited mid-interval.
    if history[-1]["hand"] != hands_played:
        history.append({"hand": hands_played, "bankroll": round(bankroll, 2)})
    return {
        "history": history,
        "hands": hands if record_hands else [],
        "final_bankroll": round(bankroll, 2),
        "max_drawdown": round(max_drawdown, 4),
        "ruined": ruined,
        "stopped_loss": stopped_loss,
        "stopped_profit": stopped_profit,
        "hands_played": hands_played,
        "stats": {
            "total_hands": total_hands,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "blackjacks": blackjacks,
            "doubles": doubles,
            "win_rate": round((wins + blackjacks) / max(total_hands, 1), 4),
        },
    }


def run_simulation(config: SimConfig) -> Dict:
    """
    Run the full Monte Carlo simulation across all active strategies.
    First sim of each strategy records per-hand data for live replay.

    Returns
    -------
    dict: {
        config: {...},
        strategies: {
            "flat": { meta: {...}, sims: [...], live_hands: [...], summary: {...} },
            ...
        }
    }
    """
    output = {
        "config": asdict(config),
        "strategies": {},
    }

    for skey in config.strategies:
        if skey not in STRATEGY_CLASSES:
            continue

        sims = []
        live_hands = []

        for i in range(config.num_sims):
            record = (i == 0)  # only record hand-by-hand for the first sim
            result = run_single_sim(config, skey, record_hands=record)

            if record:
                live_hands = result.pop("hands")
            else:
                result.pop("hands", None)

            sims.append(result)

        # Aggregate summary statistics
        finals = [s["final_bankroll"] for s in sims]
        finals_sorted = sorted(finals)
        n = len(finals)
        mean_final = sum(finals) / n
        median_final = finals_sorted[n // 2]
        variance = sum((f - mean_final) ** 2 for f in finals) / n
        std_final = variance ** 0.5
        ruin_rate = sum(1 for s in sims if s["ruined"]) / n
        stopped_loss_rate = sum(1 for s in sims if s.get("stopped_loss")) / n
        stopped_profit_rate = sum(1 for s in sims if s.get("stopped_profit")) / n
        avg_max_dd = sum(s["max_drawdown"] for s in sims) / n
        # Profit chance: % of sims where the player walked away with MORE than
        # the starting bankroll. This is the actual "did I win money?" metric.
        profit_chance = sum(1 for f in finals if f > config.start_bankroll) / n
        # EV per hand: average profit / hands actually played, across sims.
        total_profit = sum(f - config.start_bankroll for f in finals)
        total_hands_played = sum(max(s.get("hands_played", config.num_hands), 1) for s in sims)
        ev_per_hand = total_profit / max(total_hands_played, 1)

        # ── Survival-duration metrics ──
        # How many hands does the strategy typically last? Critical for the
        # progressive systems where the answer is "less than you think".
        hands_lasted = sorted(s.get("hands_played", config.num_hands) for s in sims)
        ruined_durations = sorted(
            s.get("hands_played", config.num_hands) for s in sims if s.get("ruined")
        )
        avg_hands_lasted = sum(hands_lasted) / max(len(hands_lasted), 1)
        median_hands_lasted = hands_lasted[len(hands_lasted) // 2] if hands_lasted else 0
        avg_ruin_hand = (
            sum(ruined_durations) / len(ruined_durations) if ruined_durations else None
        )
        median_ruin_hand = (
            ruined_durations[len(ruined_durations) // 2] if ruined_durations else None
        )

        def _pct(p: float) -> float:
            idx = max(0, min(n - 1, int(round(p * (n - 1)))))
            return finals_sorted[idx]

        # Resolve display name — mit_team has no class.
        display_name = (
            STRATEGY_CLASSES[skey].name if STRATEGY_CLASSES[skey] is not None
            else {"mit_team": "MIT Team"}.get(skey, skey)
        )

        # ── Build survival & profit-probability curves ──
        # Sample every num_hands/60 hands so the curves stay light to render.
        NPOINTS = 60
        grid = [round(config.num_hands * i / NPOINTS) for i in range(NPOINTS + 1)]

        def _bankroll_at(hist: List[Dict], target: int) -> float:
            """Largest history-recorded bankroll at a hand <= target."""
            br = hist[0]["bankroll"]
            for pt in hist:
                if pt["hand"] <= target:
                    br = pt["bankroll"]
                else:
                    break
            return br

        def _reached(s: Dict, target: int) -> bool:
            """Did this sim still have a live bankroll AT this hand?"""
            last_hand = s["history"][-1]["hand"]
            if s.get("ruined"):
                return last_hand > target  # ruined sim is alive only up to the hand it died
            return last_hand >= target  # SL/TP/end-of-run sim's last value carries forward

        curves = []
        for hand_target in grid:
            alive = 0
            in_profit = 0
            brs = []
            for s in sims:
                br = _bankroll_at(s["history"], hand_target)
                if _reached(s, hand_target):
                    alive += 1
                    brs.append(br)
                    if br > config.start_bankroll:
                        in_profit += 1
                else:
                    # Run already terminated — carry the last bankroll forward
                    brs.append(s["final_bankroll"])
                    if s["final_bankroll"] > config.start_bankroll:
                        in_profit += 1
            brs.sort()
            n_brs = len(brs)
            def _pct_of(arr, p):
                if not arr: return 0
                idx = max(0, min(len(arr)-1, int(round(p * (len(arr)-1)))))
                return arr[idx]
            curves.append({
                "hand": hand_target,
                "survival_pct": round(alive / n * 100, 2),
                "profit_chance_pct": round(in_profit / n * 100, 2),
                "median": round(_pct_of(brs, 0.50), 2),
                "p5":  round(_pct_of(brs, 0.05), 2),
                "p25": round(_pct_of(brs, 0.25), 2),
                "p75": round(_pct_of(brs, 0.75), 2),
                "p95": round(_pct_of(brs, 0.95), 2),
            })

        # ── Color status & interpretation tag ──
        # Color: green only if ALL of (profit_chance > 50%, median > start, ruin < 10%);
        # red if ANY of (ruin >= 25%, median <= start, drawdown >= 75%); else amber.
        median_above_start = median_final > config.start_bankroll
        if profit_chance > 0.50 and median_above_start and ruin_rate < 0.10:
            color_status = "green"
        elif ruin_rate >= 0.25 or not median_above_start or avg_max_dd >= 0.75:
            color_status = "red"
        else:
            color_status = "amber"

        # Interpretation tags — short hints flagging "why" a strategy looks
        # the way it does in the table. Multiple tags can apply.
        tags = []
        warning = None
        if ruin_rate >= 0.50:
            tags.append("Low survival")
        if mean_final > config.start_bankroll and median_final <= config.start_bankroll:
            tags.append("Outlier-driven")
            warning = "Mean inflated by rare outliers"
        if std_final > config.start_bankroll * 1.0:
            tags.append("High volatility")
        if median_final < config.start_bankroll * 0.95 and ruin_rate < 0.20:
            tags.append("Stable loser")
        if (
            median_final > config.start_bankroll * 1.05
            and avg_max_dd < 0.30
            and ruin_rate < 0.05
        ):
            tags.append("Stable winner")
        # The cheat is, by construction, an unrealistic upper bound.
        if skey == "hilo_cheat":
            tags.append("Cheating / invalid")

        summary = {
            "mean": round(mean_final, 2),
            "median": round(median_final, 2),
            "std": round(std_final, 2),
            "min": round(min(finals), 2),
            "max": round(max(finals), 2),
            "p5":  round(_pct(0.05), 2),
            "p10": round(_pct(0.10), 2),
            "p25": round(_pct(0.25), 2),
            "p75": round(_pct(0.75), 2),
            "p90": round(_pct(0.90), 2),
            "p95": round(_pct(0.95), 2),
            "ruin_rate": round(ruin_rate, 4),
            "stopped_loss_rate": round(stopped_loss_rate, 4),
            "stopped_profit_rate": round(stopped_profit_rate, 4),
            "avg_max_drawdown": round(avg_max_dd, 4),
            "roi": round((mean_final - config.start_bankroll) / config.start_bankroll * 100, 2),
            "profit_chance_pct": round(profit_chance * 100, 2),
            "ev_per_hand": round(ev_per_hand, 4),
            "median_profit": round(median_final - config.start_bankroll, 2),
            "avg_hands_lasted": round(avg_hands_lasted, 1),
            "median_hands_lasted": int(median_hands_lasted),
            "avg_ruin_hand": round(avg_ruin_hand, 1) if avg_ruin_hand is not None else None,
            "median_ruin_hand": int(median_ruin_hand) if median_ruin_hand is not None else None,
            "color_status": color_status,
            "tags": tags,
            "warning": warning,
            "curves": curves,
            "finals": [round(f, 2) for f in finals],   # for distribution chart
            # Per-sim outcome ledger drives the new distribution timeline.
            "outcomes": [
                {
                    "final": round(s["final_bankroll"], 2),
                    "hands": s.get("hands_played", config.num_hands),
                    "outcome": (
                        "ruined"  if s.get("ruined")
                        else "stopped_loss"   if s.get("stopped_loss")
                        else "stopped_profit" if s.get("stopped_profit")
                        else ("profit" if s["final_bankroll"] > config.start_bankroll
                              else ("loss" if s["final_bankroll"] < config.start_bankroll
                                    else "break_even"))
                    ),
                }
                for s in sims
            ],
        }

        # MIT team — also surface average per-role final bankrolls.
        if skey == "mit_team":
            role_finals = {role: [] for role in MIT_ROLES}
            for s in sims:
                tb = s.get("team_breakdown", {})
                for role in MIT_ROLES:
                    role_finals[role].append(tb.get(role, 0.0))
            summary["team_breakdown"] = {
                role: {
                    "mean": round(sum(vals) / max(len(vals), 1), 2),
                    "min": round(min(vals), 2) if vals else 0,
                    "max": round(max(vals), 2) if vals else 0,
                }
                for role, vals in role_finals.items()
            }
            # Average BP play rate (proportion of rounds the Big Player engaged)
            bp_rates = [s["stats"].get("bp_play_rate", 0) for s in sims]
            summary["avg_bp_play_rate"] = round(sum(bp_rates) / max(len(bp_rates), 1), 4)

        output["strategies"][skey] = {
            "meta": {
                "name": display_name,
                "color": STRATEGY_META[skey]["color"],
                "desc": STRATEGY_META[skey]["desc"],
            },
            "sims": sims,
            "live_hands": live_hands,
            "summary": summary,
        }

    return output


# ─── CLI Interface ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Blackjack Strategy Simulator")
    parser.add_argument("--decks", type=int, default=6)
    parser.add_argument("--penetration", type=float, default=75.0)
    parser.add_argument("--bankroll", type=float, default=10000)
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument("--table-min", type=float, default=25)
    parser.add_argument("--table-max", type=float, default=5000)
    parser.add_argument("--bj-pays", choices=["3:2", "6:5"], default="3:2")
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--strategies", nargs="+", default=["flat", "martingale", "hilo"],
                        choices=list(STRATEGY_CLASSES.keys()))
    parser.add_argument("--output", type=str, default="simulation_results.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stop-loss", type=float, default=None,
                        help="Terminate run when bankroll drops by this %% of start (e.g. 25)")
    parser.add_argument("--take-profit", type=float, default=None,
                        help="Terminate run when bankroll grows by this %% above start (e.g. 50)")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    config = SimConfig(
        num_decks=args.decks,
        penetration=args.penetration,
        start_bankroll=args.bankroll,
        num_hands=args.hands,
        table_min=args.table_min,
        table_max=args.table_max,
        bj_pays=args.bj_pays,
        num_sims=args.sims,
        strategies=args.strategies,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
    )

    print(f"Running {config.num_sims} simulations × {len(config.strategies)} strategies "
          f"({config.num_hands} hands each)...")

    results = run_simulation(config)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results written to {out_path}")

    # Print summary table — survival-first ranking
    print(f"\n{'Strategy':<22} {'Profit%':>9} {'Ruin%':>8} {'Median':>11} "
          f"{'EV/Hand':>10} {'MaxDD':>8} {'Tag':<22}")
    print("─" * 96)
    # Sort by profit_chance_pct descending so the best survival/profit combo is on top.
    rows = sorted(
        results["strategies"].items(),
        key=lambda kv: kv[1]["summary"].get("profit_chance_pct", 0),
        reverse=True,
    )
    for skey, data in rows:
        s = data["summary"]
        tag_str = ", ".join(s.get("tags", [])) or "—"
        print(f"{data['meta']['name']:<22} {s['profit_chance_pct']:>8.1f}% "
              f"{s['ruin_rate']*100:>7.1f}% ${s['median']:>10,.0f} "
              f"${s['ev_per_hand']:>9.3f} {s['avg_max_drawdown']*100:>7.1f}% {tag_str:<22}")
        if s.get("warning"):
            print(f"   ⚠ {s['warning']}")
        if skey == "mit_team" and "team_breakdown" in s:
            tb = s["team_breakdown"]
            print(f"   ↳ Spotter:    ${tb['spotter']['mean']:>10,.0f}   "
                  f"Controller: ${tb['controller']['mean']:>10,.0f}   "
                  f"Big Player: ${tb['big_player']['mean']:>10,.0f}   "
                  f"BP play rate: {s['avg_bp_play_rate']*100:.1f}%")


if __name__ == "__main__":
    main()

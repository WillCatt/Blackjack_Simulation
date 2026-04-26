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


STRATEGY_CLASSES = {
    "flat": FlatBet,
    "martingale": Martingale,
    "paroli": Paroli,
    "oneThreeTwoSix": OneTwoThreeSix,
    "oscar": OscarsGrind,
    "hilo": HiLoCount,
}

STRATEGY_META = {
    "flat":            {"color": "#4fc3f7", "desc": "Bet the same amount every hand. The control group."},
    "martingale":      {"color": "#ef5350", "desc": "Double after every loss, reset after a win."},
    "paroli":          {"color": "#66bb6a", "desc": "Double after a win (3-win cap), reset on a loss."},
    "oneThreeTwoSix":  {"color": "#ffa726", "desc": "Bet 1→3→2→6 units on consecutive wins."},
    "oscar":           {"color": "#ab47bc", "desc": "Raise by 1 unit after a win; reset when cycle profits 1 unit."},
    "hilo":            {"color": "#26c6da", "desc": "Bet proportional to the Hi-Lo true count."},
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
    shoe = Shoe(config.num_decks, config.penetration)
    strategy = STRATEGY_CLASSES[strategy_key](config.table_min)
    bankroll = config.start_bankroll
    max_bankroll = bankroll
    max_drawdown = 0.0
    ruined = False

    bj_multiplier = 1.5 if config.bj_pays == "3:2" else 1.2

    # Sampling for bankroll chart (cap at ~1000 points)
    sample_interval = max(1, config.num_hands // 1000)
    history = [{"hand": 0, "bankroll": bankroll}]
    hands = []

    wins, losses, pushes, blackjacks, doubles = 0, 0, 0, 0, 0

    for h in range(config.num_hands):
        if shoe.needs_reshuffle():
            shoe.reshuffle()
            strategy.on_reshuffle()

        if bankroll <= 0:
            ruined = True
            break

        # Determine bet
        raw_bet = strategy.get_bet(decks_remaining=shoe.decks_remaining)
        bet = min(max(raw_bet, config.table_min), config.table_max, bankroll)

        # Safety check — ensure enough cards
        if shoe.cards_remaining < 15:
            shoe.reshuffle()
            strategy.on_reshuffle()

        # Play the hand
        result = play_hand(shoe, config.dealer_stands_s17)

        # Calculate payout
        if result.is_blackjack and result.net_units > 0:
            payout = bet * bj_multiplier
        else:
            payout = result.net_units * bet

        bankroll = round(bankroll + payout, 2)
        max_bankroll = max(max_bankroll, bankroll)
        dd = (max_bankroll - bankroll) / max_bankroll if max_bankroll > 0 else 0
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
            if strategy_key == "hilo":
                snap["running_count"] = strategy.running_count
                snap["true_count"] = round(strategy.running_count / shoe.decks_remaining, 2)
            hands.append(snap)

    total_hands = wins + losses + pushes + blackjacks
    return {
        "history": history,
        "hands": hands if record_hands else [],
        "final_bankroll": round(bankroll, 2),
        "max_drawdown": round(max_drawdown, 4),
        "ruined": ruined,
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
        avg_max_dd = sum(s["max_drawdown"] for s in sims) / n

        output["strategies"][skey] = {
            "meta": {
                "name": STRATEGY_CLASSES[skey].name,
                "color": STRATEGY_META[skey]["color"],
                "desc": STRATEGY_META[skey]["desc"],
            },
            "sims": sims,
            "live_hands": live_hands,
            "summary": {
                "mean": round(mean_final, 2),
                "median": round(median_final, 2),
                "std": round(std_final, 2),
                "min": round(min(finals), 2),
                "max": round(max(finals), 2),
                "p10": round(finals_sorted[int(n * 0.1)], 2),
                "p90": round(finals_sorted[int(n * 0.9)], 2),
                "ruin_rate": round(ruin_rate, 4),
                "avg_max_drawdown": round(avg_max_dd, 4),
                "roi": round((mean_final - config.start_bankroll) / config.start_bankroll * 100, 2),
            },
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
    )

    print(f"Running {config.num_sims} simulations × {len(config.strategies)} strategies "
          f"({config.num_hands} hands each)...")

    results = run_simulation(config)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results written to {out_path}")

    # Print summary table
    print(f"\n{'Strategy':<20} {'Mean':>10} {'ROI':>8} {'Ruin%':>8} {'MaxDD':>8}")
    print("─" * 58)
    for skey, data in results["strategies"].items():
        s = data["summary"]
        print(f"{data['meta']['name']:<20} ${s['mean']:>9,.0f} {s['roi']:>7.1f}% "
              f"{s['ruin_rate']*100:>7.1f}% {s['avg_max_drawdown']*100:>7.1f}%")


if __name__ == "__main__":
    main()

# The Execution Game

A competition platform where you build an agent to execute a large buy order at the lowest cost.

## The Game

You have 10,000 units to buy. The market has a live order book with depth. Every time you buy, you push the price up. Other agents are executing their own orders simultaneously — some competing for the same liquidity, some creating patterns you can exploit.

**Your goal:** Fill your entire order at the lowest average cost.

**Score:** Implementation Shortfall (IS) in basis points — how much more you paid compared to the price when you started. Lower is better.

The naive approach (market order everything) gets destroyed by slippage. So you learn to slice orders, time them, read the book, and adapt to what other agents are doing.

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd vibecoding-comps
uv sync

# Test your agent locally
uv run python cli.py test

# See how the example agents perform
uv run python main.py
```

## Building Your Agent

Edit `agent/agent.py`. Your agent is a class with one method:

```python
from agents.base import BaseAgent
from engine.types import Order, OrderType, Side, TickState

class Agent(BaseAgent):
    def __init__(self, agent_id: str, target_qty: int, **kwargs):
        super().__init__(agent_id, target_qty)
        # Load config, model weights, etc.

    def on_tick(self, state: TickState) -> list[Order]:
        # state.order_book  — full book with bids/asks and depth
        # state.remaining_qty — how much you still need to buy
        # state.fills — your fills so far
        # state.trade_tape — recent trades (price, size, aggressor side)
        # state.tick / state.total_ticks — current progress
        # state.arrival_price — mid price at tick 0 (your benchmark)

        # Return a list of orders:
        # Market: Order(side=Side.BUY, size=100, order_type=OrderType.MARKET)
        # Limit:  Order(side=Side.BUY, size=100, order_type=OrderType.LIMIT, price=99.5)
        return []
```

Each tick:
1. You see the full order book and trade tape
2. You submit orders (market or limit)
3. Orders are matched against the book
4. Unfilled limit orders are cancelled at end of tick
5. Repeat for 500 ticks

## Testing Locally

```bash
# Run your agent against example bots (Naive, TWAP, VWAP)
uv run python cli.py test

# More seeds for more accurate results
uv run python cli.py test --seeds 50
```

## Competing

```bash
# First time: set up your credentials
# Create a .env file:
# COMP_SERVER=http://your-server:8000
# COMP_NAME=yourname
# COMP_TOKEN=yourtoken

# Submit your agent
uv run python cli.py submit

# Trigger a tournament
uv run python cli.py run

# Check standings
uv run python cli.py leaderboard
```

## Example Agents (in `agents/`)

Study these for inspiration:

- **NaiveAgent** — Market orders everything every tick. Gets destroyed by slippage.
- **TWAPAgent** — Equal chunks across time. Solid baseline.
- **VWAPAgent** — Volume-weighted schedule (heavier at open/close).
- **AdaptiveAgent** — Reads spread and depth, adjusts between limit and market orders.

## Strategy Ideas

- **Limit orders in the spread** — cheaper than market orders, but might not fill
- **Read the tape** — other agents' trades create patterns you can exploit
- **Urgency control** — be passive when ahead of schedule, aggressive when behind
- **Book depth analysis** — size your orders based on available liquidity
- **Almgren-Chriss** — the academic optimal execution framework
- **RL-learned policies** — train an agent offline, load weights in `__init__`

## Running the Server

For the competition host:

```bash
# Edit players.yaml with player names and tokens
# Start the server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

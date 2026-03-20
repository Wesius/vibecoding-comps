# The Execution Game

You have 10,000 units to buy. The market has a live order book. Every time you buy, you push the price up. Other players' agents are buying at the same time, competing for the same liquidity.

**Your goal:** fill your order at the lowest average cost. **Score:** how much more you paid vs the starting price (in basis points). Lower is better.

## Setup

```bash
git clone https://github.com/Wesius/vibecoding-comps.git
cd vibecoding-comps
uv sync
```

Don't have uv? Install it: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Credentials

You should have received a `.txt` file with your credentials. Rename it to `.env` and drop it in the repo root:

```bash
mv ~/Downloads/yourname.txt .env
```

## Build Your Agent

Edit `agent/agent.py`. It's pre-filled with a simple strategy. Make it better.

```python
def on_tick(self, state: TickState) -> list[Order]:
    # state.order_book  — bids/asks with depth at each price level
    # state.remaining_qty — how much you still need to buy
    # state.fills — your fills so far
    # state.trade_tape — recent trades by all participants
    # state.tick / state.total_ticks — current progress (500 ticks total)
    # state.arrival_price — the starting price (your benchmark)

    # Market order (fills immediately, but you pay the spread):
    Order(side=Side.BUY, size=100, order_type=OrderType.MARKET)

    # Limit order (cheaper if it fills, but might not):
    Order(side=Side.BUY, size=100, order_type=OrderType.LIMIT, price=99.5)
```

## Commands

```bash
# Test locally against example bots
uv run python cli.py test

# Submit your agent to the server
uv run python cli.py submit

# Run a tournament (runs all submitted agents against each other)
uv run python cli.py run

# Check the leaderboard
uv run python cli.py leaderboard
```

## Example Agents (in `agents/`)

Look at these for inspiration:

- **NaiveAgent** — Market orders everything every tick. Worst possible strategy.
- **TWAPAgent** — Equal chunks across time. Solid baseline to beat.
- **VWAPAgent** — Heavier at open/close, lighter in the middle.

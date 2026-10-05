---
layout: ../../layouts/MarkdownLayout.astro
title: 'Polymarket prediction tally'
description: 'CLI tool for keeping track of polymarket predictions'
tags: ["Python", "sql"]
---
## Introduction

Polymarket Prediction Tally is a command-line application for recording, tracking, and evaluating predictions about real-world events. The application retrieves active prediction markets from Polymarket’s Gamma Markets API, stores selected questions and user predictions in SQLite, and evaluates those predictions when the markets are resolved.

The project also includes a simulated betting system. Users can buy and sell Yes/No positions using an in-app budget, allowing them to experiment with market-style decision-making without using real money.

The application is intentionally designed as a personal analytics and learning tool. It does not execute trades or facilitate real-money gambling.

## Motivation

This project started as an exercise in building a larger, end-to-end application rather than a standalone script. I wanted to practice connecting several parts of a software system:

- an external API
- a persistent database
- a command-line user interface
- application and business logic
- automated tests

I also liked the idea of testing my own beliefs about the world. Politics, sports, and other real-world events are subjects where it is easy to form opinions without measuring how accurate they are. This application lets me make predictions about real events without spending real money, while still holding me accountable by recording the predictions and comparing them with the eventual outcomes.

The simulated budget adds another layer of accountability. A prediction can be treated not only as a Yes/No answer, but also as a decision about confidence, price, risk, and position size.

## Project Overview

The application is distributed as a Python package and exposes the `polytally` command:

```text
polytally predict <username>
polytally history <username>
polytally bet <username>
polytally sell <username>
polytally update
polytally users
```

### Prediction workflow

When a user runs `predict`, the application:

1. Finds or creates the user.
2. Fetches active Politics markets from Polymarket.
3. Updates any questions already known to the local database.
4. Displays the available questions.
5. Records the user’s Yes/No answer, timestamp, and optional explanation.

The explanation field is useful because it preserves not only what the user predicted, but also why they made the prediction at that time.

### Evaluation workflow

The `update` command checks the markets currently stored as unresolved. For markets that have resolved, it:

- updates the stored question data and outcome
- determines whether each user’s latest prediction was correct
- updates individual response records
- updates aggregate user statistics
- settles open simulated positions
- reports changes in market prices and position values

This creates a feedback loop between the original prediction and the eventual result.

### Simulated betting workflow

The betting system uses a virtual user budget. Buying a position spends budget and increases the user’s stake in either the Yes or No outcome. Selling a position returns the simulated value of that stake based on the current market price.

The position model stores separate Yes and No stakes:

```python
@dataclass
class Position:
    user_id: int
    question_id: int
    stake_yes: float
    stake_no: float
```

When a market resolves, its open positions are automatically sold at the resolved outcome prices and then removed from the active positions table.

## Code Structure

```text
polymarket_predictions_tally/
├── api.py                 # Gamma Markets API integration and parsing
├── constants.py           # Shared limits and application constants
├── initialization.py      # Database/config initialization
├── integration.py         # High-level application workflows
├── logic.py               # Domain models and core data structures
├── main.py                # Application entry point
├── utils.py               # General utility functions
├── cli/
│   ├── command.py         # Click command definitions
│   ├── prints.py          # Tables, history, statistics, and notifications
│   └── user_input.py      # Interactive prompts and input validation
├── database/
│   ├── read.py            # Read operations and database-to-model conversion
│   ├── write.py           # Inserts, updates, transactions, and settlement
│   └── utils.py           # SQL loading and position calculations
├── queries/               # SQLite schema and individual SQL queries
└── config/                # Default application configuration
```

The repository also contains example API responses and a test suite covering the API layer, database operations, CLI interactions, application logic, and integration workflows.

### Domain models: `logic.py`

The domain layer defines the objects passed between the API, CLI, and database layers. The main models are:

- `Question`: a prediction market, including its probabilities, outcomes, category, description, and resolution state
- `Event`: a Polymarket event containing multiple questions
- `User`: a local user and their simulated budget
- `Response`: a user’s prediction and optional explanation
- `Transaction`: a simulated buy or sell operation
- `Position`: the user’s aggregated Yes/No stake for a question

### Gamma Markets API: `api.py`

The API module communicates with Polymarket’s Gamma API using `requests`. It supports fetching active questions and events, retrieving individual markets, and converting raw API dictionaries into `Question` and `Event` instances.

The API returns fields such as `outcomePrices` and `outcomes` as JSON-encoded strings. The parser converts those values into Python lists and normalizes dates into timezone-aware `datetime` objects.

The API layer is separated from the rest of the application so that the rest of the system can work with typed domain objects instead of raw HTTP responses.

### Database layer

SQLite is used for local persistence. The schema contains tables for:

- users and their simulated budgets
- questions and market outcomes
- prediction responses
- prediction statistics
- simulated transactions
- open positions

SQL statements are stored as separate files under `queries/`. The application loads them through `database/utils.py`, which allows database operations to remain readable and keeps SQL separate from Python control flow.

The database layer is divided into read and write modules. Read functions return domain objects such as `User`, `Question`, `Response`, and `Position`; write functions handle inserts, updates, budget changes, and position settlement.

### Position and budget calculations

The core position calculation lives in `database/utils.py`. A transaction amount represents the simulated cash value of a buy or sell. The number of shares added or removed is calculated from the current market price:

```python
sign = 1 if transaction.transaction_type == "buy" else -1
new_stake = old_stake + sign * transaction.amount / price
budget_delta = -sign * transaction.amount
```

This means a buy decreases the available budget and a sell increases it. The calculation is shared by normal user transactions and automatic settlement when a market resolves.

### Command-line interface

The CLI is implemented with Click. `cli/command.py` defines the public commands, while `cli/user_input.py` contains interactive selection and validation logic.

The interactive interface displays market questions, end dates, existing stakes, and current Yes/No probabilities. Input is validated with Click ranges, choices, confirmation prompts, and amount limits. This keeps command definitions small while allowing the interaction logic to be tested independently.

### Application integration layer

`integration.py` coordinates complete use cases. It connects the API, database, domain models, CLI prompts, and output functions without placing all of that responsibility in any one module.

For example, the prediction workflow is intentionally short because the individual concerns are delegated to other modules:

```python
def predict(username: str, conn):
    user = get_or_make_user(conn, username)
    api_questions = api.get_questions(tag="Politics")
    update_present_questions(conn, api_questions)

    previous = get_previous_user_responses(
        conn, api_questions, user.id
    )
    question, response = process_prediction(user, api_questions, previous)

    if response is not None:
        insert_question(conn, question)
        insert_response(conn, response)
```

## Testing

The test suite is organized around the application’s layers:

- API parsing tests verify conversion of raw market data and resolution handling.
- Logic and database tests cover users, questions, responses, statistics, transactions, and positions.
- CLI tests cover prompts, amount validation, prediction flows, and betting flows.
- Integration tests exercise complete prediction, betting, selling, and database-update sessions.

Interactive behavior is tested by mocking Click prompts, while database tests use controlled SQLite connections and fixtures. This makes it possible to test the business rules without depending on a live API or real user input.

## Technologies Used

- Python 3.11
- SQLite
- Requests
- Click
- Poetry
- Pytest
- Platformdirs
- TOML configuration

## Current Scope and Future Improvements

The current version focuses on Politics markets and simulated positions. Planned improvements include richer financial statistics, tracking the purchase price of positions for more accurate profit reporting, improved question removal behavior, fuzzy/fzf-based selection, and broader manual testing against live market updates.

## Closing

Polymarket Prediction Tally combines an external data source, a local data model, an interactive interface, and automated evaluation into one small but complete application. Its main purpose is both practical and educational: it provides a structured way to test predictions about real events while demonstrating how multiple software layers work together in a maintainable Python project.

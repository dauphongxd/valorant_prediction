# Valorant Prophet Bot

Valorant Prophet is a sophisticated, multi-functional Discord bot designed for Valorant esports enthusiasts. It leverages a suite of machine learning models to predict match outcomes, integrates real-time odds from vlr.gg for a virtual betting system, and supports a multi-language user experience.

## ✨ Key Features

*   **🧠 Advanced Match Prediction:** Utilizes an ensemble of 8 different machine learning models (from Logistic Regression to TabNet and TabTransformer) to provide nuanced and accurate predictions for match outcomes.
*   **📈 Real-Time Betting Odds:** Scrapes `vlr.gg` in real-time to fetch the latest betting odds for upcoming professional matches.
*   **💰 Interactive Betting System:** A complete virtual currency system where users can place bets on upcoming matches, check their balance, and view active bets.
*   **🤖 Automated Bet Resolution:** The bot automatically monitors match results and resolves all bets, paying out winners and sending DM notifications for wins and losses.
*   **⚡ Proactive Caching:** It proactively analyzes upcoming matches and caches the prediction results, allowing for instantaneous `/predict` responses for known matchups.
*   **🏆 Server-Side Leaderboards:** A `/leaderboard` command to foster competition by displaying the top users by balance within the server.
*   **🌐 Multi-Language Support:** Fully localized with support for English and Vietnamese. Bot responses and UI elements adapt to the user's chosen language.
*   **🔒 Secure & Private:** All betting and balance commands are ephemeral (only visible to you) to protect user privacy.

## 🚀 How It Works

The bot's architecture is divided into three core components:

1.  **Data Scraping (`bet.py`):** A background task runs periodically using `Selenium` and `BeautifulSoup` to scrape `vlr.gg`. It gathers data on upcoming matches, live match statuses, and final results. This data populates the bot's internal database.
2.  **Prediction Pipeline (`main.py`):** When a prediction is requested, the bot normalizes the team names against a known database. It then runs the input through 8 different ML models, each trained on historical match data. The final prediction is a weighted average of all successful model outputs.
3.  **Discord Bot & UI (`bot.py`):** This is the user-facing component. It handles slash commands, presents data in clear embedded messages, and uses interactive components like Buttons and Modals to create a seamless betting experience. It directly interfaces with the SQLite database to manage user accounts, balances, and bets.

## 🛠️ Technology Stack

*   **Backend:** Python 3
*   **Discord API:** `discord.py`
*   **Web Scraping:** `Selenium`, `BeautifulSoup4`, `requests`
*   **Database:** `SQLite3`
*   **Machine Learning:** `CatBoost`, `LightGBM`, `XGBoost`, `Scikit-learn`, `Pytorch-Tabnet`
*   **Utilities:** `python-dotenv`, `tabulate`, `numpy`

## 🤖 Bot Commands

| Command | Description | Arguments |
| :--- | :--- | :--- |
| `/predict` | Predict the winner of a match using the ML models. | `team_a`, `team_b`, `best_of` |
| `/bet` | Fetch odds for a match and open a betting slip. | `team_a`, `team_b` |
| `/balance` | Check your wallet balance and view your active bets. | None |
| `/leaderboard` | Shows the top 10 users by balance in this server. | None |
| `/language` | Set your preferred language for bot messages. | `language` (English / Tiếng Việt) |
| `/reset` | Reset your balance to $1000 and clear all active bets. | None |
| `/help` | Shows a detailed list of all commands and how to use them. | None |
| `/stats` | **(Admin Only)** Shows bot usage statistics. | None |

## ⚙️ Setup & Installation

To run your own instance of the Valorant Prophet bot, follow these steps:

**Prerequisites:**
*   Python 3.8 or higher
*   Google Chrome installed (for the Selenium scraper)
*   A Discord Bot Token

**1. Clone the Repository**
```bash
git clone https://github.com/dauphongxd/valorant_prediction.git
cd valorant_prediction/
```

**2. Create a `requirements.txt` file**
Create a file named `requirements.txt` in the root directory and add the following dependencies:

```
discord.py
python-dotenv
selenium
webdriver-manager
beautifulsoup4
requests
tabulate
numpy
unicodedata
# --- Machine Learning Libraries ---
# Note: You may need specific versions depending on your model files.
# Pytorch is a dependency for the TabNet/TabTransformer models.
# You may need to install it separately by following the instructions on the PyTorch website: https://pytorch.org/get-started/locally/
scikit-learn
catboost
lightgbm
xgboost
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**
Create a file named `.env` in the root directory and add your bot's credentials:

```env
DISCORD_TOKEN="YOUR_DISCORD_BOT_TOKEN_HERE"

# Optional: For instantly syncing commands to a test server
GUILD_ID="YOUR_TEST_SERVER_ID_HERE"

# Your Discord User ID, for admin-only commands
DEVELOPER_ID="YOUR_DISCORD_USER_ID_HERE"
```

**5. Run the Bot**
The database (`valorant_bot.db`) will be created automatically on the first run.

```bash
python bot.py
```

## 📂 Project Structure

```
.
├── models/                     # Directory for all trained ML model artifacts
├── locales/
│   ├── en.json                 # English language strings
│   └── vi.json                 # Vietnamese language strings
├── bot.py                      # Main Discord bot logic, commands, UI, and background tasks
├── main.py                     # Machine learning pipeline, model runner, team name normalization
├── bet.py                      # All web scraping functions for vlr.gg
├── database.py                 # All SQLite database interactions and schema
├── localization.py             # Translator class for multi-language support
├── shorten_team_name.json      # Mapping for team name abbreviations
├── valid_teams.json            # List of all teams the ML models were trained on
├── .env                        # For storing secrets and environment variables
└── requirements.txt            # List of Python dependencies
```
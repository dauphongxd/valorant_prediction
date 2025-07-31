# database.py

import sqlite3
import logging
import json
import os
from datetime import datetime

DATABASE_FILE = 'valorant_bot.db'

def initialize_database():
    """Creates the database and tables if they don't exist."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # --- CORRECTED: Full table definitions ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT NOT NULL,
        balance REAL NOT NULL DEFAULT 1000.0,
        prediction_count INTEGER NOT NULL DEFAULT 0
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS bets (
        bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        match_id TEXT NOT NULL,
        team_bet_on TEXT NOT NULL,
        opponent TEXT NOT NULL,
        amount REAL NOT NULL,
        odds REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'ACTIVE',
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )''')

    # --- Tables for caching match data ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS matches (
        vlr_url TEXT PRIMARY KEY,
        team1_name TEXT NOT NULL,
        team2_name TEXT NOT NULL,
        match_time TEXT NOT NULL,
        status TEXT NOT NULL,
        last_odds_update TIMESTAMP
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS odds (
        odd_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_url TEXT NOT NULL,
        bookmaker TEXT NOT NULL,
        team1_odds REAL NOT NULL,
        team2_odds REAL NOT NULL,
        FOREIGN KEY(match_url) REFERENCES matches (vlr_url) ON DELETE CASCADE
    )''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS predictions
                   (
                       prediction_id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       team_a
                       TEXT
                       NOT
                       NULL,
                       team_b
                       TEXT
                       NOT
                       NULL,
                       best_of
                       TEXT
                       NOT
                       NULL,
                       model_version
                       TEXT
                       NOT
                       NULL,
                       winner
                       TEXT
                       NOT
                       NULL,
                       winner_prob
                       REAL
                       NOT
                       NULL,
                       successful_models
                       INTEGER
                       NOT
                       NULL,
                       total_models
                       INTEGER
                       NOT
                       NULL,
                       results_json
                       TEXT
                       NOT
                       NULL, -- Store the detailed model breakdown as JSON
                       created_at
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       UNIQUE
                   (
                       team_a,
                       team_b,
                       best_of,
                       model_version
                   )
                       )''')

    conn.commit()
    conn.close()
    logging.info("Database initialized successfully.")


# --- The rest of your database.py file is correct and needs no changes. ---
# get_user_account, place_bet, get_active_bets, etc. can all remain as they are.

def get_user_account(user_id: str, username: str):
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()
    if user is None:
        cursor.execute("INSERT INTO users (user_id, username) VALUES (?, ?)", (user_id, username))
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        user = cursor.fetchone()
        logging.info(f"Created new user account for {username} (ID: {user_id})")
    else:
        cursor.execute("UPDATE users SET username = ? WHERE user_id = ?", (username, user_id))
        conn.commit()
    conn.close()
    return user

def place_bet(user_id: str, match_id: str, team_bet_on: str, opponent: str, amount: float, odds: float):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE users SET balance = balance - ? WHERE user_id = ?", (amount, user_id))
        cursor.execute('''
                       INSERT INTO bets (user_id, match_id, team_bet_on, opponent, amount, odds)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', (user_id, match_id, team_bet_on, opponent, amount, odds))
        conn.commit()
        logging.info(f"Successfully placed bet for user {user_id} on match {match_id}")
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to place bet for user {user_id}: {e}")
        raise
    finally:
        conn.close()

def get_active_bets(user_id: str) -> list:
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM bets WHERE user_id = ? AND status = 'ACTIVE'", (user_id,))
    bets = cursor.fetchall()
    conn.close()
    return [dict(bet) for bet in bets]

def update_prediction_count(user_id: str):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET prediction_count = prediction_count + 1 WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

def upsert_matches(match_list: list):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    for match in match_list:
        cursor.execute('''
                       INSERT INTO matches (vlr_url, team1_name, team2_name, match_time, status)
                       VALUES (?, ?, ?, ?, ?) ON CONFLICT(vlr_url) DO
                       UPDATE SET
                           match_time = excluded.match_time,
                           status = excluded.status
                       ''', (match['vlr_url'], match['team1_name'], match['team2_name'], match['match_time'],
                             match['status']))
    conn.commit()
    conn.close()
    logging.info(f"Upserted {len(match_list)} matches into the database.")


def mark_matches_as_final(vlr_urls: list):
    """
    Takes a list of URLs from the results page and updates their status to 'Final',
    but ONLY if they have not already been marked as 'COMPLETED'.
    This prevents the resolution loop.
    """
    if not vlr_urls:
        return

    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    placeholders = ', '.join('?' for _ in vlr_urls)

    # --- THIS IS THE FIX ---
    # We add "AND status != 'COMPLETED'" to the query.
    query = f"UPDATE matches SET status = 'Final' WHERE vlr_url IN ({placeholders}) AND status != 'COMPLETED'"
    # -----------------------

    cursor.execute(query, vlr_urls)
    conn.commit()
    conn.close()
    logging.info(f"Marked {cursor.rowcount} new matches as 'Final' based on results page.")


def get_upcoming_match_urls():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT vlr_url FROM matches WHERE status != 'completed' AND status != 'final'")
    urls = [row[0] for row in cursor.fetchall()]
    conn.close()
    return urls

def update_match_odds(match_url: str, odds_data: list):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM odds WHERE match_url = ?", (match_url,))
        for odd in odds_data:
            cursor.execute('''
                           INSERT INTO odds (match_url, bookmaker, team1_odds, team2_odds)
                           VALUES (?, ?, ?, ?)
                           ''', (match_url, odd['bookmaker'], odd['team1_odds'], odd['team2_odds']))
        cursor.execute("UPDATE matches SET last_odds_update = ? WHERE vlr_url = ?", (datetime.now(), match_url))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to update odds for {match_url}: {e}")
    finally:
        conn.close()

def get_match_for_betting(team1_query: str, team2_query: str):
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    norm_t1 = f"%{team1_query.lower()}%"
    norm_t2 = f"%{team2_query.lower()}%"
    cursor.execute('''
                   SELECT *
                   FROM matches
                   WHERE (LOWER(team1_name) LIKE ? AND LOWER(team2_name) LIKE ?)
                      OR (LOWER(team1_name) LIKE ? AND LOWER(team2_name) LIKE ?)
                       AND status
                       != 'completed' AND status != 'final'
                   ORDER BY last_odds_update DESC LIMIT 1
                   ''', (norm_t1, norm_t2, norm_t2, norm_t1))
    match = cursor.fetchone()
    if not match:
        conn.close()
        return None, None
    cursor.execute("SELECT * FROM odds WHERE match_url = ?", (match['vlr_url'],))
    odds = cursor.fetchall()
    conn.close()
    return dict(match), [dict(o) for o in odds]

def get_matches_to_check_for_odds():
    """Fetches all matches that are still marked as 'Upcoming'."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT vlr_url FROM matches WHERE status = 'Upcoming'")
    urls = [row[0] for row in cursor.fetchall()]
    conn.close()
    return urls

def get_matches_to_resolve():
    """Finds all matches in the DB that are marked as 'Final'."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT vlr_url FROM matches WHERE status = 'Final'")
    matches = cursor.fetchall()
    conn.close()
    return [dict(match) for match in matches]


def resolve_match_bets(match_url: str, winner_name: str):
    """
    Resolves all active bets for a given match using its URL as the unique ID.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    resolved_bets_info = []

    try:
        cursor.execute("BEGIN TRANSACTION")

        # --- FIX: Query bets directly using the VLR URL as the match_id ---
        cursor.execute("SELECT * FROM bets WHERE match_id = ? AND status = 'ACTIVE'", (match_url,))
        # ------------------------------------------------------------------

        active_bets = cursor.fetchall()

        if not active_bets:
            cursor.execute("UPDATE matches SET status = 'COMPLETED' WHERE vlr_url = ?", (match_url,))
            conn.commit()
            return []

        for bet in active_bets:
            # ... (payout logic is the same) ...
            user_id = bet['user_id']
            amount_bet = bet['amount']
            odds = bet['odds']
            bet_id = bet['bet_id']
            payout = amount_bet * odds

            if bet['team_bet_on'] == winner_name:
                cursor.execute("UPDATE users SET balance = balance + ? WHERE user_id = ?", (payout, user_id))
                cursor.execute("UPDATE bets SET status = 'RESOLVED_WON' WHERE bet_id = ?", (bet_id,))
                resolved_bets_info.append(
                    {'user_id': user_id, 'outcome': 'win', 'payout': payout, 'team': bet['team_bet_on']})
            else:
                cursor.execute("UPDATE bets SET status = 'RESOLVED_LOST' WHERE bet_id = ?", (bet_id,))
                resolved_bets_info.append(
                    {'user_id': user_id, 'outcome': 'loss', 'amount': amount_bet, 'team': bet['team_bet_on']})

        cursor.execute("UPDATE matches SET status = 'COMPLETED' WHERE vlr_url = ?", (match_url,))
        conn.commit()
        logging.info(f"Successfully resolved {len(active_bets)} bets for match {match_url}.")

    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to resolve bets for match {match_url}. Transaction rolled back. Error: {e}")
        return None
    finally:
        conn.close()

    return resolved_bets_info


def save_prediction(team_a: str, team_b: str, best_of: str, model_version: str, winner: str, winner_prob: float,
                    successful_models: int, total_models: int, results: list):
    """Saves a new prediction result to the cache."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    # Convert the detailed results list to a JSON string for storage
    results_json = json.dumps(results)
    try:
        cursor.execute('''
                       INSERT INTO predictions (team_a, team_b, best_of, model_version, winner, winner_prob,
                                                successful_models, total_models, results_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''',
                       (team_a, team_b, best_of, model_version, winner, winner_prob, successful_models, total_models,
                        results_json))
        conn.commit()
        logging.info(f"Saved prediction for {team_a} vs {team_b} (Version: {model_version})")
    except sqlite3.IntegrityError:
        logging.warning(f"Prediction for {team_a} vs {team_b} (Version: {model_version}) already exists.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to save prediction: {e}")
    finally:
        conn.close()


def get_cached_prediction(team_a: str, team_b: str, best_of: str, model_version: str):
    """Retrieves a cached prediction if it exists for the current model version."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Ensure the query is symmetrical (Team A vs Team B is the same as Team B vs Team A)
    cursor.execute('''
                   SELECT *
                   FROM predictions
                   WHERE ((team_a = ? AND team_b = ?) OR (team_a = ? AND team_b = ?))
                     AND best_of = ?
                     AND model_version = ?
                   ''', (team_a, team_b, team_b, team_a, best_of, model_version))

    prediction = cursor.fetchone()
    conn.close()

    if prediction:
        # The database stores team_a and team_b as they were first entered.
        # We need to adjust the probability if the user's input is swapped.
        cached_team_a = prediction['team_a']

        # If the user's team_a is the same as the cached team_a, the data is correct.
        # If they are swapped, we must invert the probability.
        if team_a == cached_team_a:
            return dict(prediction)
        else:
            # User swapped the teams, so we adjust the output on the fly
            swapped_prediction = dict(prediction)
            swapped_prediction['winner_prob'] = 1 - prediction['winner_prob']
            # We don't need to swap the winner name because it's already correct based on the original probability.
            return swapped_prediction

    return None
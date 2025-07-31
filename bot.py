import os
import discord
from discord import app_commands, ui
from discord.ext import tasks
from dotenv import load_dotenv
from tabulate import tabulate
import asyncio
import logging
from typing import List
import json

# --- Custom Module Imports ---
from bet import (
    scrape_upcoming_matches_list,
    scrape_results_page,
    scrape_match_page_odds,
    scrape_match_winner,
    normalize_name as normalize_vlr_name
)
from main import run_all_models, normalize_team_name as normalize_model_team_name, MODEL_VERSION
import database

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Ensure the script's working directory is correct ---
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

# --- Load Environment Variables ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# --- MODIFIED get_user_account to use the database ---
# This function is now just a wrapper for the database call.
async def get_user_account(user_id: str, username: str):
    """
    Gets a user's account from the database, creating it if necessary.
    """
    # Running database operations in an executor to avoid blocking the bot's event loop
    loop = asyncio.get_running_loop()
    account = await loop.run_in_executor(None, database.get_user_account, user_id, username)
    return account


# --- In-Memory Cache for predictions (This will be addressed in a future phase) ---
prediction_cache = {}

# --- Bot Setup ---
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# --- UI Classes for Interactive Betting ---
class BettingModal(ui.Modal, title='Place Your Bet'):
    """The pop-up form where users enter their bet amount."""

    def __init__(self, team_bet_on: str, opponent: str, odds: float, match_id: str):
        super().__init__()
        self.team_bet_on = team_bet_on
        self.opponent = opponent
        self.odds = odds
        self.match_id = match_id
        self.amount_input = ui.TextInput(label=f"Amount to bet on {self.team_bet_on}", placeholder="e.g., 100 or 55.50",
                                         required=True)
        self.add_item(self.amount_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            amount = float(self.amount_input.value)
            if amount <= 0: raise ValueError("Amount must be positive.")
        except ValueError:
            await interaction.response.send_message("❌ Please enter a valid positive number.", ephemeral=True)
            return

        # Get the user's account from the DATABASE
        account = await get_user_account(str(interaction.user.id), str(interaction.user))

        if amount > account['balance']:
            await interaction.response.send_message(
                f"❌ **Insufficient Funds!** Your balance is only **${account['balance']:.2f}**.", ephemeral=True)
            return

        # NEW: Place the bet using the database function
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                database.place_bet,
                str(interaction.user.id),
                self.match_id,
                self.team_bet_on,
                self.opponent,
                amount,
                self.odds
            )

            new_balance = account['balance'] - amount
            payout = amount * self.odds
            embed = discord.Embed(title="✅ Bet Confirmed!", color=discord.Color.green(),
                                  description=f"You placed a bet on the **{self.team_bet_on}** vs **{self.opponent}** match.")
            embed.add_field(name="Your Pick", value=f"**{self.team_bet_on}**").add_field(name="Bet Amount",
                                                                                         value=f"**${amount:.2f}**").add_field(
                name="Potential Payout", value=f"**${payout:.2f}**")
            embed.set_footer(text=f"Your new balance is ${new_balance:.2f}. Good luck!")
            await interaction.response.send_message(embed=embed, ephemeral=True)

        except Exception as e:
            logging.error(f"Error during bet submission for {interaction.user}: {e}")
            await interaction.response.send_message("❌ An error occurred while placing your bet. Please try again.",
                                                    ephemeral=True)


class BettingView(ui.View):
    """The view containing the buttons for team selection."""
    # --- FIX: Now accepts the vlr_url to use as a unique ID ---
    def __init__(self, *, team1_vlr, team2_vlr, odds_t1, odds_t2, vlr_url, timeout=180):
        super().__init__(timeout=timeout)
        self.team1_vlr, self.team2_vlr = team1_vlr, team2_vlr
        self.odds_t1, self.odds_t2 = odds_t1, odds_t2
        self.match_id = vlr_url # Use the URL as the unique ID
        # -------------------------------------------------------------
        self.message = None
        self.add_item(ui.Button(label=self.team1_vlr, style=discord.ButtonStyle.primary, custom_id="team1_button"))
        self.add_item(ui.Button(label=self.team2_vlr, style=discord.ButtonStyle.secondary, custom_id="team2_button"))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        # This part doesn't need to change, it passes the self.match_id (now the URL) to the modal.
        modal = None
        if interaction.data["custom_id"] == "team1_button":
            modal = BettingModal(team_bet_on=self.team1_vlr, opponent=self.team2_vlr, odds=self.odds_t1, match_id=self.match_id)
        elif interaction.data["custom_id"] == "team2_button":
            modal = BettingModal(team_bet_on=self.team2_vlr, opponent=self.team1_vlr, odds=self.odds_t2, match_id=self.match_id)
        if modal:
            await interaction.response.send_modal(modal)
            return True
        return False

    async def on_timeout(self):
        if self.message:
            expired_embed = self.message.embeds[0]
            expired_embed.set_footer(text="This betting slip has expired.").color = discord.Color.greyple()
            for item in self.children: item.disabled = True
            await self.message.edit(embed=expired_embed, view=self)


@tasks.loop(minutes=10)
async def update_matches_cache():
    """
    The definitive background task that intelligently updates statuses from multiple sources.
    """
    logging.info("[CACHE TASK] Starting update cycle...")
    loop = asyncio.get_running_loop()

    # --- Part 1: Update statuses for UPCOMING and LIVE matches ---
    match_list = await loop.run_in_executor(None, scrape_upcoming_matches_list)
    if match_list is not None:
        await loop.run_in_executor(None, database.upsert_matches, match_list)
        logging.info("[CACHE TASK] Updated upcoming/live match statuses.")
    else:
        logging.error("[CACHE TASK] Could not scrape main matches page.")

    # --- Part 2: Update statuses for FINAL matches using the RESULTS page ---
    completed_urls = await loop.run_in_executor(None, scrape_results_page)
    if completed_urls is not None:
        await loop.run_in_executor(None, database.mark_matches_as_final, completed_urls)
    else:
        logging.error("[CACHE TASK] Could not scrape results page.")

    # --- Part 3: Check for ODDS only on UPCOMING matches ---
    matches_for_odds_check = await loop.run_in_executor(None, database.get_matches_to_check_for_odds)
    logging.info(f"[CACHE TASK] Found {len(matches_for_odds_check)} 'Upcoming' matches to check for odds.")
    for url in matches_for_odds_check:
        odds_data = await loop.run_in_executor(None, scrape_match_page_odds, url)
        if odds_data is not None:
            await loop.run_in_executor(None, database.update_match_odds, url, odds_data)
        await asyncio.sleep(2)  # Be polite

    # --- Part 4: Resolve bets for matches we know are FINAL ---
    logging.info("[CACHE TASK] Starting resolution check for 'Final' matches...")
    matches_to_resolve = await loop.run_in_executor(None, database.get_matches_to_resolve)

    if not matches_to_resolve:
        logging.info("[RESOLVER] No 'Final' matches found to resolve.")
    else:
        logging.info(f"[RESOLVER] Found {len(matches_to_resolve)} 'Final' matches to resolve.")

    for match in matches_to_resolve:
        match_url = match['vlr_url']
        winner = await loop.run_in_executor(None, scrape_match_winner, match_url)

        if winner:
            logging.info(f"[RESOLVER] Winner found for {match_url}: {winner}. Processing bets.")
            resolved_bets = await loop.run_in_executor(None, database.resolve_match_bets, match_url, winner)
            if resolved_bets is not None:
                # The DM notification logic is correct and remains here
                for bet_info in resolved_bets:
                    try:
                        user = await client.fetch_user(int(bet_info['user_id']))
                        if bet_info['outcome'] == 'win':
                            embed = discord.Embed(title="🎉 Bet Won!", color=discord.Color.gold(),
                                                  description=f"Your bet on **{bet_info['team']}** was successful!")
                            embed.add_field(name="Payout",
                                            value=f"**${bet_info['payout']:.2f}** has been added to your balance.")
                        else:
                            embed = discord.Embed(title="💔 Bet Lost", color=discord.Color.red(),
                                                  description=f"Unfortunately, your bet on **{bet_info['team']}** did not win.")
                            embed.add_field(name="Amount Lost", value=f"**${bet_info['amount']:.2f}**")
                        await user.send(embed=embed)
                    except Exception as e:
                        logging.warning(f"Failed to send bet resolution DM to {bet_info['user_id']}: {e}")

                    await asyncio.sleep(1)

        await asyncio.sleep(2)

    logging.info("[CACHE TASK] Update cycle finished.")


@client.event
async def on_ready():
    database.initialize_database()

    GUILD_ID = os.getenv('GUILD_ID')
    if GUILD_ID:
        await tree.sync(guild=discord.Object(id=GUILD_ID))
        logging.info(f"Commands synced to Guild ID: {GUILD_ID}")
    else:
        await tree.sync()
        logging.info("Commands synced globally. (May take up to an hour to update)")

    logging.info(f'Logged in as {client.user}')

    if not update_matches_cache.is_running():
        logging.info("Starting background cache update task.")
        update_matches_cache.start()

    logging.info('Bot is ready.')


# --- HELP COMMAND (No changes needed) ---
@tree.command(name="help", description="Shows how to use the bot.")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🤖 Valorant Predictor & Betting Bot Help",
        description="Use ML to predict outcomes or bet on matches with live odds.",
        color=discord.Color.purple()
    )
    embed.add_field(
        name="Automated Bet Resolution",
        value="Matches are automatically resolved and paid out shortly after they finish. You will receive a DM with the result.",
        # Changed description
        inline=False
    )
    embed.add_field(name="`/predict [team_a] [team_b] [best_of]`",
                    value="Predict a match winner using 8 different ML models.", inline=False)
    embed.add_field(name="`/bet [team_a] [team_b]`", value=(
        "Place a bet on an upcoming match.\n" "Uses live odds scraped from VLR.gg.\n" "You start with **$1000**."),
                    inline=False)
    embed.add_field(name="`/balance`", value="Check your current wallet balance and see your active bets.",
                    inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True)


# --- PREDICT COMMAND ---
@tree.command(name="predict", description="Predict the winner of a Valorant match.")
@app_commands.describe(
    team_a="Name of Team A (e.g., Cloud9 or C9)",
    team_b="Name of Team B (e.g., Sentinels or SEN)",
    best_of="The format of the match (Bo1, Bo3, or Bo5)"
)
@app_commands.choices(best_of=[
    app_commands.Choice(name="Best of 1", value="Bo1"),
    app_commands.Choice(name="Best of 3", value="Bo3"),
    app_commands.Choice(name="Best of 5", value="Bo5"),
])
async def predict_command(interaction: discord.Interaction, team_a: str, team_b: str,
                          best_of: app_commands.Choice[str]):
    await interaction.response.defer(thinking=True)

    user_team_a = normalize_model_team_name(team_a)
    user_team_b = normalize_model_team_name(team_b)

    if not user_team_a or not user_team_b:
        await interaction.followup.send(
            "❌ Invalid team name provided for prediction. Please use a valid name or abbreviation found in `main.py`.")
        return

    # NEW: Get user account and update prediction count in the database
    account = await get_user_account(str(interaction.user.id), str(interaction.user))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, database.update_prediction_count, str(interaction.user.id))

    cached_prediction = await loop.run_in_executor(
        None,
        database.get_cached_prediction,
        user_team_a,
        user_team_b,
        best_of.value,
        MODEL_VERSION  # Check against the current model version
    )

    if cached_prediction:
        logging.info(f"Cache hit for {user_team_a} vs {user_team_b}. Serving from database.")
        # Re-create the embed from cached data
        winner = cached_prediction['winner']
        winner_prob = cached_prediction['winner_prob']

        embed = discord.Embed(title="📈 Valorant Match Prediction (Cached)",
                              description=f"## ⚔️ **{user_team_a}** vs **{user_team_b}**", color=discord.Color.blue())
        embed.add_field(name="🏆 Overall Prediction (Weighted Avg)",
                        value=f"**Predicted Winner: `{winner}`**\nConfidence: `{winner_prob:.2%}`", inline=False)

        # Unpack the detailed results from JSON
        results = json.loads(cached_prediction['results_json'])
        headers = ["Model", "Winner", "Confidence"]
        for tier, title in {'⭐': "⭐ Standard", '⭐⭐': "⭐⭐ Advanced", '⭐⭐⭐': "⭐⭐⭐ Deep Learning"}.items():
            # ... (the logic to build the table from 'results' is identical to the non-cached path)
            # This can be refactored into a helper function to avoid code duplication
            tier_data = []
            for res in results:
                if res['tier'] == tier:
                    if res['prob_a_wins'] is not None:
                        model_winner = user_team_a if res['prob_a_wins'] >= 0.5 else user_team_b
                        model_conf = f"{res['prob_a_wins'] if res['prob_a_wins'] >= 0.5 else 1 - res['prob_a_wins']:.2%}"
                        tier_data.append([res['model'], model_winner, model_conf])
                    else:
                        tier_data.append([res['model'], "Failed", "N/A"])
            if tier_data:
                table_string = tabulate(tier_data, headers=headers, tablefmt="github")
                embed.add_field(name=title, value=f"```\n{table_string}\n```", inline=False)

        embed.set_footer(
            text=f"Result from cache (Model Version: {MODEL_VERSION}). Aggregated from {cached_prediction['successful_models']}/{cached_prediction['total_models']} models.")
        await interaction.followup.send(embed=embed)
        return  # End execution here

    logging.info(f"Cache miss for {user_team_a} vs {user_team_b}. Running models.")
    results = run_all_models(user_team_a, user_team_b, best_of.value)

    # --- The rest of the prediction logic is unchanged for now ---
    results = run_all_models(user_team_a, user_team_b, best_of.value)
    tier_weights = {'⭐': 1.0, '⭐⭐': 1.5, '⭐⭐⭐': 2.0}
    total_weighted_prob, total_weight, successful_models_count = 0, 0, 0

    for res in results:
        if res['prob_a_wins'] is not None:
            weight = tier_weights.get(res['tier'], 1.0)
            total_weighted_prob += res['prob_a_wins'] * weight
            total_weight += weight
            successful_models_count += 1

    if total_weight == 0:
        await interaction.followup.send(embed=discord.Embed(title="Prediction Failed",
                                                            description="None of the models could generate a prediction.",
                                                            color=discord.Color.red()))
        return

    avg_prob = total_weighted_prob / total_weight
    winner = user_team_a if avg_prob > 0.5 else user_team_b
    winner_prob = avg_prob if winner == user_team_a else 1 - avg_prob

    await loop.run_in_executor(
        None,
        database.save_prediction,
        user_team_a,
        user_team_b,
        best_of.value,
        MODEL_VERSION,
        winner,
        winner_prob,
        successful_models_count,
        len(results),
        results  # Pass the detailed results for JSON storage
    )

    embed = discord.Embed(title="📈 Valorant Match Prediction",
                          description=f"## ⚔️ **{user_team_a}** vs **{user_team_b}**", color=discord.Color.blue())
    embed.add_field(name="🏆 Overall Prediction (Weighted Avg)",
                    value=f"**Predicted Winner: `{winner}`**\nConfidence: `{winner_prob:.2%}`", inline=False)

    headers = ["Model", "Winner", "Confidence"]
    for tier, title in {'⭐': "⭐ Standard", '⭐⭐': "⭐⭐ Advanced", '⭐⭐⭐': "⭐⭐⭐ Deep Learning"}.items():
        tier_data = []
        for res in results:
            if res['tier'] == tier:
                if res['prob_a_wins'] is not None:
                    model_winner = user_team_a if res['prob_a_wins'] >= 0.5 else user_team_b
                    model_conf = f"{res['prob_a_wins'] if res['prob_a_wins'] >= 0.5 else 1 - res['prob_a_wins']:.2%}"
                    tier_data.append([res['model'], model_winner, model_conf])
                else:
                    tier_data.append([res['model'], "Failed", "N/A"])
        if tier_data:
            table_string = tabulate(tier_data, headers=headers, tablefmt="github")
            embed.add_field(name=title, value=f"```\n{table_string}\n```", inline=False)

    embed.set_footer(text=f"Aggregated from {successful_models_count}/{len(results)} successful models.")
    await interaction.followup.send(embed=embed)


# --- BALANCE COMMAND (MODIFIED) ---
@tree.command(name="balance", description="Check your wallet and view your active bets.")
async def balance_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    # Get user account and bets from the DATABASE
    loop = asyncio.get_running_loop()
    user_id_str = str(interaction.user.id)
    account = await get_user_account(user_id_str, str(interaction.user))
    active_bets = await loop.run_in_executor(None, database.get_active_bets, user_id_str)

    embed = discord.Embed(title=f"💰 {interaction.user.display_name}'s Wallet", color=discord.Color.green())
    embed.add_field(name="Current Balance", value=f"**${account['balance']:.2f}**", inline=False)

    if active_bets:
        bet_list = []
        for i, bet in enumerate(active_bets):
            payout = bet['amount'] * bet['odds']
            bet_list.append(
                f"`{i + 1}.` **${bet['amount']:.2f}** on **{bet['team_bet_on']}** vs {bet['opponent']} (Odds: {bet['odds']:.2f}) for **${payout:.2f}**")
        embed.add_field(name="Active Bets", value="\n".join(bet_list), inline=False)
    else:
        embed.add_field(name="Active Bets", value="You have no active bets.", inline=False)

    await interaction.followup.send(embed=embed)


# --- BET COMMAND (No logic changes, but now interacts with new database functions) ---
@tree.command(name="bet", description="Fetch odds for a match and open a betting slip.")
@app_commands.describe(
    team_a="The name of the first team.",
    team_b="The name of the second team."
)
async def bet_command(interaction: discord.Interaction, team_a: str, team_b: str):
    await interaction.response.defer(thinking=True)
    loop = asyncio.get_running_loop()
    match, odds_list = await loop.run_in_executor(None, database.get_match_for_betting, team_a, team_b)

    if not match:
        await interaction.followup.send(
            f"❌ Could not find an upcoming match between '{team_a}' and '{team_b}' in my cache.", ephemeral=True)
        return

    if not odds_list:
        await interaction.followup.send(
            f"🟡 Match found for **{match['team1_name']} vs {match['team2_name']}**, but no betting odds are listed yet.",
            ephemeral=True)
        return

    # --- FIX: No longer averaging. We take the odds from the first bookmaker in the list. ---
    first_odds = odds_list[0]
    odds_t1 = first_odds['team1_odds']
    odds_t2 = first_odds['team2_odds']
    bookmaker = first_odds['bookmaker']
    # -----------------------------------------------------------------------------------

    vlr_team1 = match['team1_name']
    vlr_team2 = match['team2_name']

    embed = discord.Embed(
        title=f"Betting Slip: {vlr_team1} vs {vlr_team2}",
        description="Click a team below to place your bet. This slip will expire in 3 minutes.",
        color=discord.Color.gold()
    )
    embed.add_field(name=f"{vlr_team1} Odds", value=f"**{odds_t1:.2f}**")
    embed.add_field(name=f"{vlr_team2} Odds", value=f"**{odds_t2:.2f}**")
    # Update footer to show which bookmaker's odds are being used.
    embed.set_footer(text=f"Odds from {bookmaker} (via local cache).")

    view = BettingView(
        team1_vlr=vlr_team1,
        team2_vlr=vlr_team2,
        odds_t1=odds_t1,
        odds_t2=odds_t2,
        vlr_url=match['vlr_url']
    )
    message = await interaction.followup.send(embed=embed, view=view)
    view.message = message


# --- Run the Bot ---
client.run(TOKEN)
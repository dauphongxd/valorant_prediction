import os
import asyncio
import logging  # <-- Standard library imports first
import json
from typing import List
import datetime

# --- SETUP LOGGING ---
# This MUST be the first thing after standard library imports.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import discord  # <-- Third-party imports next
from discord import app_commands, ui
from discord.ext import tasks
from dotenv import load_dotenv
from tabulate import tabulate

from localization import translator # <-- Your local/project imports last

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


# --- Ensure the script's working directory is correct ---
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

# --- Load Environment Variables ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
DEVELOPER_ID = os.getenv('DEVELOPER_ID')

# --- EMOJI LOAD UP ----
ASCENDANT_EMOJI = "<:Ascendant_Valorant:1401039199322374207>"
IMMORTAL_EMOJI  = "<:Valorant_Immortal_3:1401039197422227537>"
RADIANT_EMOJI   = "<:Radiant_Valorant:1401039189897777172>"

# --- MODIFIED get_user_account to use the database ---
# This function is now just a wrapper for the database call.
async def get_user_account(user_id: str, username: str, guild_id: str = None):
    """
    Gets a user's account from the database, creating it if necessary,
    and ensures the user is linked to the guild.
    """
    # Running database operations in an executor to avoid blocking the bot's event loop
    loop = asyncio.get_running_loop()
    # Now 'guild_id' is a known variable that can be passed to the database function.
    account = await loop.run_in_executor(None, database.get_user_account, user_id, username, guild_id)
    return account


# --- In-Memory Cache for predictions (This will be addressed in a future phase) ---
prediction_cache = {}

# --- Bot Setup ---
intents = discord.Intents.default()
intents.members = True  # <--- ADD THIS ONE LINE
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# --- UI Classes for Interactive Betting ---
class BettingModal(ui.Modal):  # <-- Remove the static title from here
    # <-- Add user_locale to the constructor
    def __init__(self, team_bet_on: str, opponent: str, odds: float, match_id: str, user_locale: str):
        super().__init__(title=translator.get_string("betting_modal.modal_title", user_locale))
        self.team_bet_on = team_bet_on
        self.opponent = opponent
        self.odds = odds
        self.match_id = match_id
        self.amount_input = ui.TextInput(
            label=translator.get_string("betting_modal.amount_label", user_locale, team_bet_on=self.team_bet_on),
            placeholder=translator.get_string("betting_modal.amount_placeholder", user_locale),
            required=True
        )
        self.add_item(self.amount_input)

    async def on_submit(self, interaction: discord.Interaction):
        # 1. Get user account and language
        account = await get_user_account(str(interaction.user.id), str(interaction.user))
        user_locale = account['language'] if account else 'en'

        # 2. Validate the bet amount
        try:
            amount = float(self.amount_input.value)
            if amount <= 0: raise ValueError("Amount must be positive.")
        except ValueError:
            await interaction.response.send_message(
                translator.get_string("betting_modal.invalid_amount_error", user_locale), ephemeral=True)
            return

        # 3. Re-fetch account to ensure the balance is current
        account = await get_user_account(str(interaction.user.id), str(interaction.user))

        # 4. Check for sufficient funds
        if amount > account['balance']:
            await interaction.response.send_message(
                translator.get_string("betting_modal.insufficient_funds_error", user_locale,
                                      balance=account['balance']),
                ephemeral=True)
            return

        # 5. --- THIS IS THE FIX ---
        # Ensure the interaction is from a server (guild) and not a DM.
        if not interaction.guild:
            await interaction.response.send_message(
                translator.get_string("betting_modal.dm_error", user_locale), # Use the new translation key
                ephemeral=True
            )
            return
        # --- END OF FIX ---

        # 6. Proceed with placing the bet
        try:
            loop = asyncio.get_running_loop()
            guild_id = str(interaction.guild.id)  # This line is now safe
            await loop.run_in_executor(
                None,
                database.place_bet,
                str(interaction.user.id), guild_id, self.match_id, self.team_bet_on,
                self.opponent, amount, self.odds
            )

            new_balance = account['balance'] - amount
            payout = amount * self.odds

            # Build and send the confirmation embed
            embed = discord.Embed(
                title=translator.get_string("betting_modal.bet_confirmed_title", user_locale),
                color=discord.Color.green(),
                description=translator.get_string("betting_modal.bet_confirmed_desc", user_locale,
                                                  team_bet_on=self.team_bet_on, opponent=self.opponent)
            )
            embed.add_field(name=translator.get_string("betting_modal.your_pick_field", user_locale),
                            value=f"**{self.team_bet_on}**")
            embed.add_field(name=translator.get_string("betting_modal.bet_amount_field", user_locale),
                            value=f"**${amount:.2f}**")
            embed.add_field(name=translator.get_string("betting_modal.payout_field", user_locale),
                            value=f"**${payout:.2f}**")
            embed.set_footer(
                text=translator.get_string("betting_modal.new_balance_footer", user_locale, new_balance=new_balance))
            await interaction.response.send_message(embed=embed, ephemeral=True)

        except ValueError as e:
            if "Insufficient funds" in str(e):
                # Re-fetch the absolute latest balance to show the user
                final_account = await get_user_account(str(interaction.user.id), str(interaction.user))
                await interaction.response.send_message(
                    translator.get_string("betting_modal.insufficient_funds_error", user_locale,
                                          balance=final_account['balance']),
                    ephemeral=True)
            else:
                # Handle other potential ValueErrors, like from bad input
                await interaction.response.send_message(
                    translator.get_string("betting_modal.invalid_amount_error", user_locale), ephemeral=True)

        except Exception as e:
            logging.error(f"Error during bet submission for {interaction.user}: {e}")
            await interaction.response.send_message(translator.get_string("betting_modal.generic_error", user_locale),
                                                    ephemeral=True)


class BettingView(ui.View):
    # <-- Add user_locale to the constructor
    def __init__(self, *, team1_vlr, team2_vlr, odds_t1, odds_t2, vlr_url, user_locale, timeout=180):
        super().__init__(timeout=timeout)
        self.team1_vlr, self.team2_vlr = team1_vlr, team2_vlr
        self.odds_t1, self.odds_t2 = odds_t1, odds_t2
        self.match_id = vlr_url
        self.user_locale = user_locale  # <-- Store the language
        self.message = None
        self.add_item(ui.Button(label=self.team1_vlr, style=discord.ButtonStyle.primary, custom_id="team1_button"))
        self.add_item(ui.Button(label=self.team2_vlr, style=discord.ButtonStyle.secondary, custom_id="team2_button"))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        modal = None
        # <-- Pass the stored language to the BettingModal
        if interaction.data["custom_id"] == "team1_button":
            modal = BettingModal(team_bet_on=self.team1_vlr, opponent=self.team2_vlr, odds=self.odds_t1,
                                 match_id=self.match_id, user_locale=self.user_locale)
        elif interaction.data["custom_id"] == "team2_button":
            modal = BettingModal(team_bet_on=self.team2_vlr, opponent=self.team1_vlr, odds=self.odds_t2,
                                 match_id=self.match_id, user_locale=self.user_locale)

        if modal:
            await interaction.response.send_modal(modal)
            return True
        return False

    async def on_timeout(self):
        if self.message:
            expired_embed = self.message.embeds[0]
            # <-- Translate the timeout footer
            expired_embed.set_footer(text=translator.get_string("betting_view.expired_footer", self.user_locale))
            expired_embed.color = discord.Color.greyple()
            for item in self.children: item.disabled = True
            await self.message.edit(embed=expired_embed, view=self)


class ResetConfirmationView(ui.View):
    def __init__(self, *, user_locale: str, timeout=30):
        super().__init__(timeout=timeout)
        self.value = None # This will store whether the user confirmed or cancelled

        # Dynamically set the labels for the decorator-defined buttons
        # The decorators will handle creating and adding the buttons.
        self.children[0].label = translator.get_string("reset_command.confirm_button_label", user_locale)
        self.children[1].label = translator.get_string("reset_command.cancel_button_label", user_locale)

    # The decorator creates the button and adds it to the view.
    # We set the style and a unique custom_id here.
    @ui.button(style=discord.ButtonStyle.danger, custom_id="confirm_reset")
    async def confirm_button_callback(self, interaction: discord.Interaction, button: ui.Button):
        self.value = True
        self.stop() # Stop the view from listening to more interactions
        # We need to respond to THIS interaction to prevent an "Interaction failed" error
        await interaction.response.defer()

    # The decorator for the second button.
    @ui.button(style=discord.ButtonStyle.secondary, custom_id="cancel_reset")
    async def cancel_button_callback(self, interaction: discord.Interaction, button: ui.Button):
        self.value = False
        self.stop()
        await interaction.response.defer()


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

        # --- NEW LOGIC TO HANDLE 404 ---
        if odds_data == "404_NOT_FOUND":
            # The match page doesn't exist, so remove it from our database.
            await loop.run_in_executor(None, database.remove_match_by_url, url)
        elif odds_data is not None and isinstance(odds_data, list):
            # If we got a list of odds (and not None), update them.
            await loop.run_in_executor(None, database.update_match_odds, url, odds_data)
        # If odds_data is None (due to a temporary error), we do nothing and will simply try again on the next 10-minute cycle.

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
                        # First, get the user's full account data from the DB
                        account = await loop.run_in_executor(None, database.get_user_account, bet_info['user_id'],
                                                             "Unknown")
                        # Get their language, defaulting to 'en' if something goes wrong
                        user_locale = account['language'] if account else 'en'

                        user = await client.fetch_user(int(bet_info['user_id']))
                        if bet_info['outcome'] == 'win':
                            embed = discord.Embed(
                                title=translator.get_string("bet_resolution_dm.win_title", user_locale),
                                color=discord.Color.gold(),
                                description=translator.get_string("bet_resolution_dm.win_desc", user_locale,
                                                                  team=bet_info['team'])
                            )
                            embed.add_field(
                                name=translator.get_string("bet_resolution_dm.win_payout_field", user_locale),
                                value=translator.get_string("bet_resolution_dm.win_payout_value", user_locale,
                                                            payout=bet_info['payout'])
                            )
                        else:  # Loss
                            embed = discord.Embed(
                                title=translator.get_string("bet_resolution_dm.loss_title", user_locale),
                                color=discord.Color.red(),
                                description=translator.get_string("bet_resolution_dm.loss_desc", user_locale,
                                                                  team=bet_info['team'])
                            )
                            embed.add_field(
                                name=translator.get_string("bet_resolution_dm.loss_amount_field", user_locale),
                                value=translator.get_string("bet_resolution_dm.loss_amount_value", user_locale,
                                                            amount=bet_info['amount'])
                            )
                        await user.send(embed=embed)
                    except Exception as e:
                        logging.warning(f"Failed to send bet resolution DM to {bet_info['user_id']}: {e}")

                    await asyncio.sleep(1)

        await asyncio.sleep(2)

    logging.info("[PROACTIVE TASK] Checking for matches that need predictions.")
    matches_to_predict = await loop.run_in_executor(None, database.get_matches_for_proactive_prediction, MODEL_VERSION)
    if not matches_to_predict:
        logging.info("[PROACTIVE TASK] All upcoming matches have predictions.")
    else:
        logging.info(f"[PROACTIVE TASK] Found {len(matches_to_predict)} new matches to analyze.")
        for match in matches_to_predict:
            team_a = normalize_model_team_name(match['team1_name'])
            team_b = normalize_model_team_name(match['team2_name'])
            match_url = match['vlr_url']
            scraped_format = match.get('best_of_format', 'N/A')

            if not team_a or not team_b:
                logging.warning(f"[PROACTIVE TASK] Could not normalize team names for {match_url}. Skipping.")
                continue

            # --- NEW DIAGNOSTIC LOGGING ---
            logging.info("--- [PROACTIVE CHECK] ---")
            logging.info(f"  > Match URL: {match_url}")
            logging.info(f"  > Scraped Format from DB: '{scraped_format}' (Type: {type(scraped_format)})")
            # --- END LOGGING -

            # --- CORE FALLBACK LOGIC ---
            # Ideal Case: The format was scraped successfully.
            if scraped_format and scraped_format != 'N/A':
                logging.info(f"  > Decision: Format is VALID. Running ONE prediction for {scraped_format}.")
                logging.info(f"[PROACTIVE TASK] Analyzing {team_a} vs {team_b} for specific format: {scraped_format}.")
                try:
                    # Run models just once for the specific format
                    results = await loop.run_in_executor(None, run_all_models, team_a, team_b, scraped_format)

                    # (This aggregation logic is the same)
                    tier_weights = {'⭐': 1.0, '⭐⭐': 1.5, '⭐⭐⭐': 2.0}
                    total_weighted_prob, total_weight = 0, 0
                    for res in results:
                        if res['prob_a_wins'] is not None:
                            weight = tier_weights.get(res['tier'], 1.0)
                            total_weighted_prob += res['prob_a_wins'] * weight
                            total_weight += weight

                    if total_weight > 0:
                        avg_prob = total_weighted_prob / total_weight
                        winner = team_a if avg_prob > 0.5 else team_b
                        winner_prob = avg_prob if winner == team_a else 1 - avg_prob
                        await loop.run_in_executor(
                            None, database.save_proactive_prediction, match_url, scraped_format, MODEL_VERSION, winner,
                            winner_prob, team_a, team_b, results
                        )
                        logging.info(f"  > Saved proactive prediction for {match_url} ({scraped_format}).")
                    else:
                        logging.warning(f"  > All models failed for {match_url}. Cannot save prediction.")
                except Exception as e:
                    logging.error(f"[PROACTIVE TASK] An error occurred running models for {match_url}: {e}")

            # Fallback Case: The format is 'N/A', so we run all three.
            else:
                logging.info(f"  > Decision: Format is INVALID or 'N/A'. Triggering FALLBACK for Bo3")
                for best_of in ['Bo3']:
                    try:
                        results = await loop.run_in_executor(None, run_all_models, team_a, team_b, best_of)

                        tier_weights = {'⭐': 1.0, '⭐⭐': 1.5, '⭐⭐⭐': 2.0}
                        total_weighted_prob, total_weight = 0, 0
                        for res in results:
                            if res['prob_a_wins'] is not None:
                                weight = tier_weights.get(res['tier'], 1.0)
                                total_weighted_prob += res['prob_a_wins'] * weight
                                total_weight += weight

                        if total_weight > 0:
                            avg_prob = total_weighted_prob / total_weight
                            winner = team_a if avg_prob > 0.5 else team_b
                            winner_prob = avg_prob if winner == team_a else 1 - avg_prob
                            await loop.run_in_executor(
                                None, database.save_proactive_prediction, match_url, best_of, MODEL_VERSION, winner,
                                winner_prob, team_a, team_b, results
                            )
                            logging.info(f"  > Saved fallback prediction for {best_of} format.")
                        else:
                            logging.warning(f"  > All models failed for {best_of} format. Cannot save prediction.")

                        await asyncio.sleep(5)  # Polite wait between formats
                    except Exception as e:
                        logging.error(
                            f"[PROACTIVE TASK] An error occurred in fallback for {team_a} vs {team_b} ({best_of}): {e}")

            await asyncio.sleep(10)  # Polite wait between matches


    logging.info("[CACHE TASK] Performing database cleanup for stale matches.")
    await loop.run_in_executor(None, database.remove_stale_upcoming_matches, 5) # Clean matches older than 5 days

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
    # Defer the response first
    await interaction.response.defer(ephemeral=True)

    # 1. Get the user's full account from the database
    account = await get_user_account(str(interaction.user.id), str(interaction.user))

    # 2. Get their saved language
    user_locale = account['language'] if account else 'en'

    embed = discord.Embed(
        title=translator.get_string("help_command.title", user_locale),
        description=translator.get_string("help_command.description", user_locale),
        color=discord.Color.purple()
    )
    embed.add_field(
        name=translator.get_string("help_command.resolution_field_title", user_locale),
        value=translator.get_string("help_command.resolution_field_value", user_locale),
        inline=False
    )
    embed.add_field(
        name=translator.get_string("help_command.predict_field_title", user_locale),
        value=translator.get_string("help_command.predict_field_value", user_locale),
        inline=False
    )
    embed.add_field(
        name=translator.get_string("help_command.bet_field_title", user_locale),
        value=translator.get_string("help_command.bet_field_value", user_locale),
        inline=False
    )
    embed.add_field(
        name=translator.get_string("help_command.balance_field_title", user_locale),
        value=translator.get_string("help_command.balance_field_value", user_locale),
        inline=False
    )
    embed.add_field(
        name=translator.get_string("help_command.leaderboard_field_title", user_locale),
        value=translator.get_string("help_command.leaderboard_field_value", user_locale),
        inline=False
    )
    embed.add_field(
        name=translator.get_string("help_command.language_field_title", user_locale),
        value=translator.get_string("help_command.language_field_value", user_locale),
        inline=False
    )
    embed.add_field(
        name=translator.get_string("help_command.reset_field_title", user_locale),
        value=translator.get_string("help_command.reset_field_value", user_locale),
        inline=False
    )
    await interaction.followup.send(embed=embed)


@client.event
async def on_guild_join(guild: discord.Guild):
    """
    Attempts to find the user who added the bot and sends them a private welcome message.
    """
    logging.info(f"Joined new guild: {guild.name} (ID: {guild.id})")

    inviter = None
    # --- The Audit Log Workaround ---
    if guild.me.guild_permissions.view_audit_log:
        try:
            async for entry in guild.audit_logs(action=discord.AuditLogAction.bot_add, limit=5):
                if entry.target.id == client.user.id:
                    if (datetime.datetime.now(datetime.timezone.utc) - entry.created_at).total_seconds() < 30:
                        inviter = entry.user
                        break
        except discord.Forbidden:
            logging.warning(f"Missing 'View Audit Log' permission in {guild.name} to find the inviter.")
        except Exception as e:
            logging.error(f"Error while fetching audit log for {guild.name}: {e}")

    if not inviter:
        logging.warning(f"Could not determine who invited the bot to {guild.name}.")
        return

    # --- Build the bilingual embed ---
    en_desc = (
        f'{translator.get_string("welcome_message.description", "en")}\n\n'
        f'• **`/help`**: {translator.get_string("welcome_message.help_intro", "en")}\n'
        f'• **`/language`**: {translator.get_string("welcome_message.language_intro", "en")}'
    )
    vi_desc = (
        f'{translator.get_string("welcome_message.description", "vi")}\n\n'
        f'• **`/help`**: {translator.get_string("welcome_message.help_intro", "vi")}\n'
        f'• **`/language`**: {translator.get_string("welcome_message.language_intro", "vi")}'
    )

    embed = discord.Embed(
        title=translator.get_string("welcome_message.title", "en"),
        color=discord.Color.purple()
    )
    embed.add_field(name="Welcome!", value=en_desc, inline=False)
    embed.add_field(name="Chào Mừng!", value=vi_desc, inline=False)

    # --- NEW: Create the text for the support link ---
    support_link_text = (
        f'Join Support Server:\n'
        "https://discord.gg/J76HkBDP2U"
    )

    # --- Send the messages as a DM ---
    try:
        # First, send the embed
        await inviter.send(embed=embed)
        # Then, send the support link as a separate, normal message
        await inviter.send(support_link_text)

        logging.info(f"Successfully sent welcome DM to {inviter.name} for joining {guild.name}.")
    except discord.Forbidden:
        logging.warning(f"Failed to send welcome DM to {inviter.name}. They may have DMs disabled.")
    except Exception as e:
        logging.error(f"An unexpected error occurred when sending welcome DM to {inviter.name}: {e}")


# --- PREDICT COMMAND ---
@tree.command(name="predict", description="Predict the winner of a Valorant match.")
@app_commands.describe(
    team_a="Name of Team A (e.g., Cloud9 or C9)",
    team_b="Name of Team B (e.g., Sentinels or SEN)",
    best_of="The format of the match (e.g., Best of 3)"
)
@app_commands.choices(best_of=[
    # The 'name' is what the user sees, the 'value' is what the bot gets.
    # The 'value' now matches our standard format.
    app_commands.Choice(name="Best of 1", value="Bo1"),
    app_commands.Choice(name="Best of 3", value="Bo3"),
    app_commands.Choice(name="Best of 5", value="Bo5"),
])
async def predict_command(interaction: discord.Interaction, team_a: str, team_b: str,
                          best_of: app_commands.Choice[str]):
    await interaction.response.defer(thinking=True)

    guild_id = str(interaction.guild.id) if interaction.guild else None
    account = await get_user_account(str(interaction.user.id), str(interaction.user), guild_id)
    user_locale = account['language'] if account else 'en'

    user_team_a = normalize_model_team_name(team_a)
    user_team_b = normalize_model_team_name(team_b)

    if not user_team_a or not user_team_b:
        error_msg = translator.get_string("predict_command.invalid_team_error", user_locale, team_a=team_a,
                                          team_b=team_b)
        await interaction.followup.send(error_msg)
        return

    tier_icons = {'⭐': ASCENDANT_EMOJI, '⭐⭐': IMMORTAL_EMOJI, '⭐⭐⭐': RADIANT_EMOJI}
    loop = asyncio.get_running_loop()

    # --- NEW: Step 1 - Try to find a pre-calculated prediction ---
    proactive_prediction = None
    match_for_pred = await loop.run_in_executor(None, database.find_upcoming_match_by_teams, user_team_a, user_team_b)

    if match_for_pred:
        logging.info(f"Found upcoming match {match_for_pred['vlr_url']} for prediction. Checking proactive cache.")
        proactive_prediction = await loop.run_in_executor(
            None, database.get_proactive_prediction, match_for_pred['vlr_url'], best_of.value, MODEL_VERSION
        )

    if proactive_prediction:
        logging.info(f"PROACTIVE cache hit for {user_team_a} vs {user_team_b}. Serving from database.")
        await loop.run_in_executor(None, database.update_prediction_count, str(interaction.user.id))

        results = json.loads(proactive_prediction['results_json'])

        # The winner and their win probability are stored authoritatively in the database.
        # We should use them directly for the final result, not recalculate them.
        winner = proactive_prediction['winner']
        winner_prob = proactive_prediction['winner_prob']

        # We only need to adjust the individual model results for the display table if the user's
        # team order is different from the order that was used when the prediction was stored.
        if user_team_a != proactive_prediction['team_a']:
            for res in results:
                if res['prob_a_wins'] is not None:
                    # Flip the probability because the user's perspective of Team A is different.
                    res['prob_a_wins'] = 1.0 - res['prob_a_wins']

        # 'winner' and 'winner_prob' now hold the correct, authoritative values,
        # and 'results' is adjusted correctly for the display table.
        # The rest of the embed building logic can now proceed without issue.

        successful_models_count = sum(1 for res in results if res['prob_a_wins'] is not None)

        embed = discord.Embed(title=translator.get_string("predict_command.embed_title", user_locale),
                              description=translator.get_string("predict_command.embed_description", user_locale,
                                                                user_team_a=user_team_a, user_team_b=user_team_b),
                              color=discord.Color.blue())
        embed.add_field(name=translator.get_string("predict_command.overall_prediction_title", user_locale),
                        value=translator.get_string("predict_command.overall_prediction_value", user_locale,
                                                    winner=winner, winner_prob=winner_prob), inline=False)
        headers = [translator.get_string("predict_command.model_header_model", user_locale),
                   translator.get_string("predict_command.model_header_winner", user_locale),
                   translator.get_string("predict_command.model_header_confidence", user_locale)]
        tier_titles = {'⭐': translator.get_string("predict_command.tier_standard_title", user_locale),
                       '⭐⭐': translator.get_string("predict_command.tier_advanced_title", user_locale),
                       '⭐⭐⭐': translator.get_string("predict_command.tier_deep_learning_title", user_locale)}

        for tier, title in tier_titles.items():
            tier_data = []
            for res in results:
                if res['tier'] == tier:
                    if res['prob_a_wins'] is not None:
                        model_winner = user_team_a if res['prob_a_wins'] >= 0.5 else user_team_b
                        model_conf = f"{res['prob_a_wins'] if res['prob_a_wins'] >= 0.5 else 1 - res['prob_a_wins']:.2%}"
                        tier_data.append([res['model'], model_winner, model_conf])
                    else:
                        tier_data.append(
                            [res['model'], translator.get_string("predict_command.model_failed", user_locale),
                             translator.get_string("predict_command.model_na", user_locale)])
            if tier_data:
                table_string = tabulate(tier_data, headers=headers, tablefmt="github")
                embed.add_field(name=f"{tier_icons[tier]} {title}", value=f"```\n{table_string}\n```", inline=False)

        embed.set_footer(text=f"⚡ Instant prediction from proactive analysis (Model v{MODEL_VERSION}).")
        await interaction.followup.send(embed=embed)
        return
    # --- END of proactive prediction logic ---

    # --- Fallback: Step 2 - Check the user-request cache (original behavior) ---
    await loop.run_in_executor(None, database.update_prediction_count, str(interaction.user.id))
    cached_prediction = await loop.run_in_executor(
        None, database.get_cached_prediction, user_team_a, user_team_b, best_of.value, MODEL_VERSION
    )

    if cached_prediction:
        logging.info(f"USER cache hit for {user_team_a} vs {user_team_b}. Serving from database.")
        results = json.loads(cached_prediction['results_json'])
        # Adjust probabilities if user input is swapped vs stored
        if user_team_a != cached_prediction['team_a']:
            for res in results:
                if res['prob_a_wins'] is not None:
                    res['prob_a_wins'] = 1.0 - res['prob_a_wins']

        # Recalculate the aggregate winner based on the possibly swapped results
        tier_weights = {'⭐': 1.0, '⭐⭐': 1.5, '⭐⭐⭐': 2.0}
        total_weighted_prob, total_weight = 0, 0
        for res in results:
            if res['prob_a_wins'] is not None:
                weight = tier_weights.get(res['tier'], 1.0)
                total_weighted_prob += res['prob_a_wins'] * weight
                total_weight += weight

        if total_weight > 0:
            avg_prob = total_weighted_prob / total_weight
            winner = user_team_a if avg_prob > 0.5 else user_team_b
            winner_prob = avg_prob if winner == user_team_a else 1 - avg_prob
        else:  # Should not happen if there is a cached prediction, but as a safe guard
            winner = "N/A"
            winner_prob = 0

        successful_models_count = cached_prediction['successful_models']
        total_models = cached_prediction['total_models']

        embed = discord.Embed(title=translator.get_string("predict_command.embed_title", user_locale),
                              description=translator.get_string("predict_command.embed_description", user_locale,
                                                                user_team_a=user_team_a, user_team_b=user_team_b),
                              color=discord.Color.blue())
        embed.add_field(name=translator.get_string("predict_command.overall_prediction_title", user_locale),
                        value=translator.get_string("predict_command.overall_prediction_value", user_locale,
                                                    winner=winner, winner_prob=winner_prob), inline=False)
        headers = [translator.get_string("predict_command.model_header_model", user_locale),
                   translator.get_string("predict_command.model_header_winner", user_locale),
                   translator.get_string("predict_command.model_header_confidence", user_locale)]
        tier_titles = {'⭐': translator.get_string("predict_command.tier_standard_title", user_locale),
                       '⭐⭐': translator.get_string("predict_command.tier_advanced_title", user_locale),
                       '⭐⭐⭐': translator.get_string("predict_command.tier_deep_learning_title", user_locale)}

        for tier, title in tier_titles.items():
            tier_data = []
            for res in results:
                if res['tier'] == tier:
                    if res['prob_a_wins'] is not None:
                        model_winner = user_team_a if res['prob_a_wins'] >= 0.5 else user_team_b
                        model_conf = f"{res['prob_a_wins'] if res['prob_a_wins'] >= 0.5 else 1 - res['prob_a_wins']:.2%}"
                        tier_data.append([res['model'], model_winner, model_conf])
                    else:
                        tier_data.append(
                            [res['model'], translator.get_string("predict_command.model_failed", user_locale),
                             translator.get_string("predict_command.model_na", user_locale)])
            if tier_data:
                table_string = tabulate(tier_data, headers=headers, tablefmt="github")
                embed.add_field(name=f"{tier_icons[tier]} {title}", value=f"```\n{table_string}\n```", inline=False)

        embed.set_footer(
            text=f"Result from request cache (Model v{MODEL_VERSION}). Aggregated from {successful_models_count}/{total_models} models.")
        await interaction.followup.send(embed=embed)
        return

    # --- Fallback: Step 3 - Run models on-demand (original behavior) ---
    logging.info(f"NO cache hit for {user_team_a} vs {user_team_b}. Running models on-demand.")
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
        embed = discord.Embed(title=translator.get_string("predict_command.prediction_failed_title", user_locale),
                              description=translator.get_string("predict_command.prediction_failed_desc", user_locale),
                              color=discord.Color.red())
        await interaction.followup.send(embed=embed)
        return

    avg_prob = total_weighted_prob / total_weight
    winner = user_team_a if avg_prob > 0.5 else user_team_b
    winner_prob = avg_prob if winner == user_team_a else 1 - avg_prob

    # Save to the user-request cache for next time
    await loop.run_in_executor(None, database.save_prediction, user_team_a, user_team_b, best_of.value, MODEL_VERSION,
                               winner, winner_prob, successful_models_count, len(results), results)

    embed = discord.Embed(title=translator.get_string("predict_command.embed_title", user_locale),
                          description=translator.get_string("predict_command.embed_description", user_locale,
                                                            user_team_a=user_team_a, user_team_b=user_team_b),
                          color=discord.Color.blue())
    embed.add_field(name=translator.get_string("predict_command.overall_prediction_title", user_locale),
                    value=translator.get_string("predict_command.overall_prediction_value", user_locale, winner=winner,
                                                winner_prob=winner_prob), inline=False)
    headers = [translator.get_string("predict_command.model_header_model", user_locale),
               translator.get_string("predict_command.model_header_winner", user_locale),
               translator.get_string("predict_command.model_header_confidence", user_locale)]
    tier_titles = {'⭐': translator.get_string("predict_command.tier_standard_title", user_locale),
                   '⭐⭐': translator.get_string("predict_command.tier_advanced_title", user_locale),
                   '⭐⭐⭐': translator.get_string("predict_command.tier_deep_learning_title", user_locale)}

    for tier, title in tier_titles.items():
        tier_data = []
        for res in results:
            if res['tier'] == tier:
                if res['prob_a_wins'] is not None:
                    model_winner = user_team_a if res['prob_a_wins'] >= 0.5 else user_team_b
                    model_conf = f"{res['prob_a_wins'] if res['prob_a_wins'] >= 0.5 else 1 - res['prob_a_wins']:.2%}"
                    tier_data.append([res['model'], model_winner, model_conf])
                else:
                    tier_data.append([res['model'], translator.get_string("predict_command.model_failed", user_locale),
                                      translator.get_string("predict_command.model_na", user_locale)])
        if tier_data:
            table_string = tabulate(tier_data, headers=headers, tablefmt="github")
            embed.add_field(name=f"{tier_icons[tier]} {title}", value=f"```\n{table_string}\n```", inline=False)

    embed.set_footer(text=translator.get_string("predict_command.embed_footer", user_locale,
                                                successful_models_count=successful_models_count,
                                                len_results=len(results)))
    await interaction.followup.send(embed=embed)


# --- BALANCE COMMAND (MODIFIED) ---
@tree.command(name="balance", description="Check your wallet and view your active bets.")
async def balance_command(interaction: discord.Interaction):
    # Defer the response first
    await interaction.response.defer(ephemeral=True)

    # 1. Get the user's full account from the database
    guild_id = str(interaction.guild.id) if interaction.guild else None
    account = await get_user_account(str(interaction.user.id), str(interaction.user), guild_id)

    # 2. Get their saved language, defaulting to 'en' if something goes wrong
    user_locale = account['language'] if account else 'en'

    # Get user account and bets from the DATABASE
    loop = asyncio.get_running_loop()
    user_id_str = str(interaction.user.id)
    account = await get_user_account(user_id_str, str(interaction.user))
    active_bets = await loop.run_in_executor(None, database.get_active_bets, user_id_str)

    embed = discord.Embed(
        title=translator.get_string("balance_command.embed_title", user_locale,
                                    display_name=interaction.user.display_name),
        color=discord.Color.green()
    )
    embed.add_field(
        name=translator.get_string("balance_command.balance_field_title", user_locale),
        value=f"**${account['balance']:.2f}**",
        inline=False
    )
    active_bets_title = translator.get_string("balance_command.active_bets_field_title", user_locale)

    if active_bets:
        bet_list = []
        for i, bet in enumerate(active_bets):
            payout = bet['amount'] * bet['odds']
            bet_list.append(
                f"`{i + 1}.` **${bet['amount']:.2f}** on **{bet['team_bet_on']}** vs {bet['opponent']} (Odds: {bet['odds']:.2f}) for **${payout:.2f}**")
        embed.add_field(name=active_bets_title, value="\n".join(bet_list), inline=False)
    else:
        embed.add_field(
            name=active_bets_title,
            value=translator.get_string("balance_command.no_active_bets", user_locale),
            inline=False
        )

    await interaction.followup.send(embed=embed)


# --- BET COMMAND (No logic changes, but now interacts with new database functions) ---
@tree.command(name="bet", description="Fetch odds for a match and open a betting slip.")
@app_commands.describe(
    team_a="The name or abbreviation of the first team (e.g., LOUD or LLL).",
    team_b="The name or abbreviation of the second team (e.g., Gen.G or GENG)."
)
async def bet_command(interaction: discord.Interaction, team_a: str, team_b: str):
    await interaction.response.defer(ephemeral=True)

    guild_id = str(interaction.guild.id) if interaction.guild else None
    account = await get_user_account(str(interaction.user.id), str(interaction.user), guild_id)
    user_locale = account['language'] if account else 'en'

    loop = asyncio.get_running_loop()
    match, odds_list = None, None

    # --- NEW TWO-STEP LOOKUP LOGIC ---

    # Attempt 1: Search using the user's raw input. This is fast and catches most matches.
    logging.info(f"[Bet Command] Attempt 1: Searching for match with raw input: '{team_a}' vs '{team_b}'")
    match, odds_list = await loop.run_in_executor(None, database.get_match_for_betting, team_a, team_b)

    # Attempt 2: If the raw search fails, try normalizing the names using the JSON file.
    if not match:
        logging.info(f"[Bet Command] Attempt 1 failed. Normalizing names for Attempt 2.")
        norm_team_a = normalize_model_team_name(team_a)
        norm_team_b = normalize_model_team_name(team_b)

        # Only proceed if BOTH names could be successfully normalized
        if norm_team_a and norm_team_b:
            logging.info(
                f"[Bet Command] Attempt 2: Searching with normalized names: '{norm_team_a}' vs '{norm_team_b}'")
            match, odds_list = await loop.run_in_executor(None, database.get_match_for_betting, norm_team_a,
                                                          norm_team_b)
    # ------------------------------------

    # Now, check if we found a match after EITHER attempt
    if not match:
        await interaction.followup.send(
            translator.get_string("bet_command.no_match_found_error", user_locale, team_a=team_a, team_b=team_b),
            ephemeral=True)
        return

    if not odds_list:
        await interaction.followup.send(
            translator.get_string("bet_command.no_odds_found_error", user_locale, team1_name=match['team1_name'],
                                  team2_name=match['team2_name']),
            ephemeral=True)
        return

    # The rest of the command is unchanged and will now work with the found match.
    first_odds = odds_list[0]
    odds_t1 = first_odds['team1_odds']
    odds_t2 = first_odds['team2_odds']
    bookmaker = first_odds['bookmaker']
    vlr_team1 = match['team1_name']
    vlr_team2 = match['team2_name']

    embed = discord.Embed(
        title=translator.get_string("bet_command.slip_title", user_locale, vlr_team1=vlr_team1, vlr_team2=vlr_team2),
        description=translator.get_string("bet_command.slip_description", user_locale),
        color=discord.Color.gold()
    )
    embed.add_field(name=translator.get_string("bet_command.odds_field_title", user_locale, team_name=vlr_team1),
                    value=f"**{odds_t1:.2f}**")
    embed.add_field(name=translator.get_string("bet_command.odds_field_title", user_locale, team_name=vlr_team2),
                    value=f"**{odds_t2:.2f}**")
    embed.set_footer(text=translator.get_string("bet_command.slip_footer", user_locale, bookmaker=bookmaker))

    view = BettingView(
        team1_vlr=vlr_team1,
        team2_vlr=vlr_team2,
        odds_t1=odds_t1,
        odds_t2=odds_t2,
        vlr_url=match['vlr_url'],
        user_locale=user_locale
    )
    message = await interaction.followup.send(embed=embed, view=view)
    view.message = message


# --- LANGUAGE COMMAND ---
@tree.command(name="language", description="Set your preferred language for bot notifications and messages.")
@app_commands.describe(language="Choose the language you want the bot to use for you.")
@app_commands.choices(language=[
    app_commands.Choice(name="English", value="en"),
    app_commands.Choice(name="Tiếng Việt", value="vi"),
])
async def language_command(interaction: discord.Interaction, language: app_commands.Choice[str]):
    """Allows a user to save their language preference to the database."""
    await interaction.response.defer(ephemeral=True)

    user_id_str = str(interaction.user.id)
    chosen_lang_code = language.value
    chosen_lang_name = language.name

    # Check if the chosen language is actually supported by our translator
    if chosen_lang_code not in translator.locales:
        # We get the locale from the interaction to make sure this error message itself is translated
        user_locale = str(interaction.locale)
        await interaction.followup.send(translator.get_string("language_command.unsupported_language", user_locale))
        return

    # Run the database update in the background thread
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        database.set_user_language,
        user_id_str,
        chosen_lang_code
    )

    # Respond with a success message IN THEIR NEWLY CHOSEN LANGUAGE for immediate feedback
    success_text = translator.get_string("language_command.success_message", chosen_lang_code, lang_name=chosen_lang_name)
    await interaction.followup.send(success_text)


# --- RESET COMMAND ---
@tree.command(name="reset", description="Reset your balance to $1000 and clear all active bets.")
async def reset_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    # Get the user's saved language for the confirmation message
    guild_id = str(interaction.guild.id) if interaction.guild else None
    account = await get_user_account(str(interaction.user.id), str(interaction.user), guild_id)
    user_locale = account['language'] if account else 'en'

    # Create the confirmation view and embed
    view = ResetConfirmationView(user_locale=user_locale)

    embed = discord.Embed(
        title=translator.get_string("reset_command.confirmation_title", user_locale),
        description=translator.get_string("reset_command.confirmation_desc", user_locale),
        color=discord.Color.orange()
    )

    # Send the confirmation message and wait for the user's response
    await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    await view.wait()  # Wait until a button is clicked or the view times out

    # Disable buttons on the original message after interaction
    for item in view.children:
        item.disabled = True
    await interaction.edit_original_response(view=view)

    # --- Process the result ---
    if view.value is True:
        # User confirmed the reset
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, database.reset_user_account, str(interaction.user.id))

            # Send a success message
            success_text = translator.get_string("reset_command.success_message", user_locale)
            await interaction.followup.send(success_text, ephemeral=True)
        except Exception as e:
            # Handle potential database errors
            logging.error(f"Error during account reset for {interaction.user.id}: {e}")
            await interaction.followup.send("❌ An error occurred while resetting your account.", ephemeral=True)

    elif view.value is False:
        # User cancelled
        cancel_text = translator.get_string("reset_command.cancel_message", user_locale)
        await interaction.followup.send(cancel_text, ephemeral=True)

    else:  # This happens if the view times out (view.value is None)
        timeout_text = translator.get_string("reset_command.timeout_message", user_locale)
        await interaction.followup.send(timeout_text, ephemeral=True)


# --- LEADERBOARD COMMAND ---

@tree.command(name="leaderboard", description="Shows the top users by balance in this server.")
async def leaderboard_command(interaction: discord.Interaction):
    await interaction.response.defer()

    # Get the user's language preference
    account = await get_user_account(str(interaction.user.id), str(interaction.user))
    user_locale = account['language'] if account else 'en'

    # This command only works in a server
    if not interaction.guild:
        await interaction.followup.send(translator.get_string("leaderboard_command.dm_error", user_locale))
        return

    # Fetch leaderboard data from the database
    guild_id = str(interaction.guild.id)
    loop = asyncio.get_running_loop()
    leaderboard_data = await loop.run_in_executor(None, database.get_leaderboard_for_guild, guild_id)

    # Create the embed
    embed = discord.Embed(
        title=translator.get_string("leaderboard_command.embed_title", user_locale, server_name=interaction.guild.name),
        color=discord.Color.gold()
    )

    if not leaderboard_data:
        embed.description = translator.get_string("leaderboard_command.no_one_on_board", user_locale)
    else:
        leaderboard_entries = []
        rank_emojis = ["🥇", "🥈", "🥉"]
        for i, user in enumerate(leaderboard_data):
            rank = rank_emojis[i] if i < 3 else f"**`{i + 1}.`**"
            username = user['username']
            balance = user['balance']
            leaderboard_entries.append(f"{rank} **{username}** - `${balance:,.2f}`")

        embed.description = "\n".join(leaderboard_entries)

    embed.set_footer(text=translator.get_string("leaderboard_command.footer_text", user_locale))

    await interaction.followup.send(embed=embed)

# --- STATS COMMAND ---
@tree.command(name="stats", description="Shows bot statistics (Admin only).")
async def stats_command(interaction: discord.Interaction):
    """A private command to show database statistics, aware of context."""

    # Admin check
    if str(interaction.user.id) != DEVELOPER_ID:
        await interaction.response.send_message("❌ You do not have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    loop = asyncio.get_running_loop()

    # --- THIS IS THE CONTEXT-AWARE LOGIC ---
    if interaction.guild is None:
        # We are in a DM, get global stats
        stats = await loop.run_in_executor(None, database.get_global_database_stats)
        embed_title = "📊 Global Bot Statistics"
        embed_desc = "Live data from across all servers."
    else:
        # We are in a server, get guild-specific stats
        guild_id = str(interaction.guild.id)
        stats = await loop.run_in_executor(None, database.get_guild_database_stats, guild_id)
        embed_title = f"📊 Statistics for {interaction.guild.name}"
        embed_desc = "Live data for this server only."
    # ----------------------------------------

    if stats is None:
        await interaction.followup.send("❌ An error occurred while fetching statistics.", ephemeral=True)
        return

    embed = discord.Embed(
        title=embed_title,
        description=embed_desc,
        color=discord.Color.blue()
    )
    embed.add_field(name="Total Users", value=f"👥 {stats['total_users']}", inline=True)
    embed.add_field(name="Active Bets", value=f"🎟️ {stats['active_bets']}", inline=True)
    embed.add_field(name="Total Money in Active Bets", value=f"💵 ${stats['total_money']:.2f}", inline=True)

    await interaction.followup.send(embed=embed, ephemeral=True)

# --- Run the Bot ---
client.run(TOKEN)
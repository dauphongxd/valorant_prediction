# bot.py (With fix for KeyError: 'balance')

import os
import discord
from discord import app_commands, ui
from dotenv import load_dotenv
from tabulate import tabulate
import json
import asyncio
from typing import List

# --- Custom Module Imports ---
from bet import get_vlr_odds, normalize_name as normalize_vlr_name
from main import run_all_models, normalize_team_name as normalize_model_team_name, TEAM_NAME_MAPPING

# --- Ensure the script's working directory is correct ---
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

# --- Load Environment Variables ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
DEVELOPER_ID = os.getenv('DEVELOPER_USER_ID')

if not TOKEN:
    raise ValueError("DISCORD_TOKEN not found in .env file. Please create one.")
if not DEVELOPER_ID:
    print("WARNING: DEVELOPER_USER_ID not set. /stats and /resolve commands will be unusable.")

# --- User Stats Persistence ---
STATS_FILE = 'user_stats.json'
stats_lock = asyncio.Lock()
user_stats = {}

def load_user_stats():
    global user_stats
    try:
        with open(STATS_FILE, 'r') as f:
            user_stats = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user_stats = {}

async def save_user_stats():
    async with stats_lock:
        with open(STATS_FILE, 'w') as f:
            json.dump(user_stats, f, indent=4)


async def get_user_account(user_id: str, username: str):
    """
    Gets a user's account, creating or updating it as needed to ensure
    all necessary keys ('balance', 'active_bets') are present.
    """
    async with stats_lock:
        # Check if the user is completely new
        if user_id not in user_stats:
            user_stats[user_id] = {
                'username': username,
                'count': 0,
                'balance': 1000.0,
                'active_bets': []
            }
        else:
            # If user exists, ensure all keys are present for backward compatibility.
            # This prevents the KeyError.
            account = user_stats[user_id]
            if 'balance' not in account:
                account['balance'] = 1000.0
            if 'active_bets' not in account:
                account['active_bets'] = []
            # Also good practice to update their username in case they changed it
            account['username'] = username

    return user_stats[user_id]


# --- In-Memory Cache for predictions ---
prediction_cache = {}

# --- Bot Setup ---
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# --- Developer Check ---
def is_developer():
    def predicate(interaction: discord.Interaction) -> bool:
        return str(interaction.user.id) == DEVELOPER_ID

    return app_commands.check(predicate)


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

        # The get_user_account call is now safe
        account = await get_user_account(str(interaction.user.id), str(interaction.user))

        if amount > account['balance']:
            await interaction.response.send_message(
                f"❌ **Insufficient Funds!** Your balance is only **${account['balance']:.2f}**.", ephemeral=True)
            return

        account['balance'] -= amount
        account['active_bets'].append(
            {'match_id': self.match_id, 'team_bet_on': self.team_bet_on, 'opponent': self.opponent, 'amount': amount,
             'odds': self.odds})
        await save_user_stats()

        payout = amount * self.odds
        embed = discord.Embed(title="✅ Bet Confirmed!", color=discord.Color.green(),
                              description=f"You placed a bet on the **{self.team_bet_on}** vs **{self.opponent}** match.")
        embed.add_field(name="Your Pick", value=f"**{self.team_bet_on}**").add_field(name="Bet Amount",
                                                                                     value=f"**${amount:.2f}**").add_field(
            name="Potential Payout", value=f"**${payout:.2f}**")
        embed.set_footer(text=f"Your new balance is ${account['balance']:.2f}. Good luck!")
        await interaction.response.send_message(embed=embed, ephemeral=True)


class BettingView(ui.View):
    """The view containing the buttons for team selection."""

    def __init__(self, *, team1_vlr, team2_vlr, odds_t1, odds_t2, timeout=180):
        super().__init__(timeout=timeout)
        self.team1_vlr, self.team2_vlr = team1_vlr, team2_vlr
        self.odds_t1, self.odds_t2 = odds_t1, odds_t2
        self.match_id = f"{normalize_vlr_name(team1_vlr)}-vs-{normalize_vlr_name(team2_vlr)}"
        self.message = None
        self.add_item(ui.Button(label=self.team1_vlr, style=discord.ButtonStyle.primary, custom_id="team1_button"))
        self.add_item(ui.Button(label=self.team2_vlr, style=discord.ButtonStyle.secondary, custom_id="team2_button"))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        modal = None
        if interaction.data["custom_id"] == "team1_button":
            modal = BettingModal(team_bet_on=self.team1_vlr, opponent=self.team2_vlr, odds=self.odds_t1,
                                 match_id=self.match_id)
        elif interaction.data["custom_id"] == "team2_button":
            modal = BettingModal(team_bet_on=self.team2_vlr, opponent=self.team1_vlr, odds=self.odds_t2,
                                 match_id=self.match_id)
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


@client.event
async def on_ready():
    load_user_stats()
    # If you are still having sync issues, make sure your ID is here. Otherwise, you can comment this out.
    GUILD_ID = os.getenv('GUILD_ID')  # Recommended to add GUILD_ID to your .env file
    if GUILD_ID:
        await tree.sync(guild=discord.Object(id=GUILD_ID))
        print(f"Commands synced to Guild ID: {GUILD_ID}")
    else:
        await tree.sync()  # Global sync
        print("Commands synced globally. (May take up to an hour to update)")

    print(f'Logged in as {client.user}')
    print('Bot is ready.')


# --- HELP COMMAND ---
@tree.command(name="help", description="Shows how to use the bot.")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🤖 Valorant Predictor & Betting Bot Help",
        description="Use ML to predict outcomes or bet on matches with live odds.",
        color=discord.Color.purple()
    )
    embed.add_field(
        name="`/predict [team_a] [team_b] [best_of]`",
        value="Predict a match winner using 8 different ML models.",
        inline=False
    )
    embed.add_field(
        name="`/bet [team_a] [team_b] [team_to_bet_on] [amount]`",
        value=(
            "Place a bet on an upcoming match.\n"
            "Uses live odds scraped from VLR.gg.\n"
            "You start with **$1000**."
        ),
        inline=False
    )
    embed.add_field(
        name="`/balance`",
        value="Check your current wallet balance and see your active bets.",
        inline=False
    )
    embed.add_field(
        name="`/resolve [team_a] [team_b] [winning_team]` (Developer Only)",
        value="Settles a match's bets and pays out winners.",
        inline=False
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)


# --- STATS COMMAND (Developer-Only) ---
@tree.command(name="stats", description="View bot usage statistics (Developer only).")
@is_developer()
async def stats_command(interaction: discord.Interaction):
    total_predictions = sum(data.get('count', 0) for data in user_stats.values())
    total_users = len(user_stats)
    total_bets = sum(len(data.get('active_bets', [])) for data in user_stats.values())

    embed = discord.Embed(
        title="📊 Bot Usage Statistics",
        description=f"**Total Predictions:** {total_predictions}\n**Total Active Bets:** {total_bets}\n**Unique Users:** {total_users}",
        color=discord.Color.blue()
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)


@stats_command.error
async def stats_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CheckFailure):
        await interaction.response.send_message("This command is restricted to the bot developer.", ephemeral=True)
    else:
        await interaction.response.send_message("An unexpected error occurred.", ephemeral=True)


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

    # Get/create user account and increment prediction count
    account = await get_user_account(str(interaction.user.id), str(interaction.user))
    account['count'] += 1
    await save_user_stats()

    # Run model predictions (this part is from your original bot.py)
    results = run_all_models(user_team_a, user_team_b, best_of.value)

    # Calculate weighted average
    tier_weights = {'⭐': 1.0, '⭐⭐': 1.5, '⭐⭐⭐': 2.0}
    total_weighted_prob = 0
    total_weight = 0
    successful_models_count = 0

    for res in results:
        prob_a_wins = res['prob_a_wins']
        if prob_a_wins is not None:
            weight = tier_weights.get(res['tier'], 1.0)
            total_weighted_prob += prob_a_wins * weight
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

    embed = discord.Embed(title="📈 Valorant Match Prediction",
                          description=f"## ⚔️ **{user_team_a}** vs **{user_team_b}**", color=discord.Color.blue())
    embed.add_field(name="🏆 Overall Prediction (Weighted Avg)",
                    value=f"**Predicted Winner: `{winner}`**\nConfidence: `{winner_prob:.2%}`", inline=False)

    # Create tables for results
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


# --- BALANCE COMMAND ---
@tree.command(name="balance", description="Check your wallet and view your active bets.")
async def balance_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    account = await get_user_account(str(interaction.user.id), str(interaction.user))

    embed = discord.Embed(title=f"💰 {interaction.user.display_name}'s Wallet", color=discord.Color.green())
    embed.add_field(name="Current Balance", value=f"**${account['balance']:.2f}**", inline=False)

    if account['active_bets']:
        bet_list = []
        for i, bet in enumerate(account['active_bets']):
            payout = bet['amount'] * bet['odds']
            bet_list.append(
                f"`{i + 1}.` **${bet['amount']:.2f}** on **{bet['team_bet_on']}** vs {bet['opponent']} (Odds: {bet['odds']:.2f}) for **${payout:.2f}**")
        embed.add_field(name="Active Bets", value="\n".join(bet_list), inline=False)
    else:
        embed.add_field(name="Active Bets", value="You have no active bets.", inline=False)

    await interaction.followup.send(embed=embed)


# --- BET COMMAND ---
@tree.command(name="bet", description="Fetch odds for a match and open a betting slip.")
@app_commands.describe(
    team_a="The name of the first team.",
    team_b="The name of the second team."
)
async def bet_command(interaction: discord.Interaction, team_a: str, team_b: str):
    await interaction.response.defer(thinking=True)

    # 1. Scrape odds using the imported function
    loop = asyncio.get_running_loop()
    vlr_data = await loop.run_in_executor(None, get_vlr_odds, team_a, team_b)

    # 2. Handle Scraper Results
    if vlr_data.get('error'):
        error_map = {
            'match_not_found': f"❌ Could not find an upcoming match between '{team_a}' and '{team_b}' on VLR.gg.",
            'no_odds_listed': f"🟡 Match found, but no betting odds are listed yet.",
            'page_scrape_failed': "❌ An error occurred trying to scrape the match page.",
            'selenium_error': "❌ A browser automation error occurred. The developer has been notified."
        }
        await interaction.followup.send(error_map.get(vlr_data.get('error', 'unknown')), ephemeral=True)
        return

    # 3. Calculate Average Odds
    avg_odds_t1 = sum(o['team1_odds'] for o in vlr_data['odds']) / len(vlr_data['odds'])
    avg_odds_t2 = sum(o['team2_odds'] for o in vlr_data['odds']) / len(vlr_data['odds'])
    vlr_team1 = vlr_data['team1_vlr']
    vlr_team2 = vlr_data['team2_vlr']

    # 4. Create the initial embed and the button view
    embed = discord.Embed(
        title=f"Betting Slip: {vlr_team1} vs {vlr_team2}",
        description="Click a team below to place your bet. This slip will expire in 3 minutes.",
        color=discord.Color.gold()
    )
    embed.add_field(name=f"{vlr_team1} Odds", value=f"**{avg_odds_t1:.2f}**")
    embed.add_field(name=f"{vlr_team2} Odds", value=f"**{avg_odds_t2:.2f}**")
    embed.set_footer(text="Odds are an average from available bookmakers on VLR.gg.")

    # Instantiate the view with the necessary data
    view = BettingView(
        team1_vlr=vlr_team1,
        team2_vlr=vlr_team2,
        odds_t1=avg_odds_t1,
        odds_t2=avg_odds_t2
    )

    # 5. Send the message with the embed and buttons
    message = await interaction.followup.send(embed=embed, view=view)
    view.message = message  # Store the message in the view so we can edit it on timeout.


# --- Run the Bot ---
client.run(TOKEN)

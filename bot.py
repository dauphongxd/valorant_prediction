# bot.py (Updated with developer-only stats command)

import os
import discord
from discord import app_commands
from dotenv import load_dotenv
from tabulate import tabulate
import json
import asyncio

# --- Ensure the script's working directory is correct ---
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

# --- Import your existing ML pipeline function ---
from main import run_all_models

# --- Load Environment Variables ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
# NEW: Load the Developer's User ID from the .env file
DEVELOPER_ID = os.getenv('DEVELOPER_USER_ID')

if not TOKEN:
    raise ValueError("DISCORD_TOKEN not found in .env file. Please create one.")
if not DEVELOPER_ID:
    print("WARNING: DEVELOPER_USER_ID not set in .env file. The /stats command will be unusable.")

# --- User Stats Persistence ---
STATS_FILE = 'user_stats.json'
stats_lock = asyncio.Lock()
user_stats = {}

def load_user_stats():
    """Loads user stats from the JSON file into memory."""
    global user_stats
    try:
        with open(STATS_FILE, 'r') as f:
            user_stats = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user_stats = {}

async def save_user_stats():
    """Saves the current user stats from memory to the JSON file."""
    async with stats_lock:
        with open(STATS_FILE, 'w') as f:
            json.dump(user_stats, f, indent=4)

# --- In-Memory Cache for match predictions ---
prediction_cache = {}

# --- Bot Setup ---
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# --- NEW: Custom Check for Developer-Only Commands ---
def is_developer():
    """A custom check to see if the user is the bot developer."""
    def predicate(interaction: discord.Interaction) -> bool:
        # Check if the person using the command has the ID from the .env file
        return str(interaction.user.id) == DEVELOPER_ID
    return app_commands.check(predicate)

@client.event
async def on_ready():
    load_user_stats()
    await tree.sync()
    print(f'Logged in as {client.user} (ID: {client.user.id})')
    print(f'Loaded {len(user_stats)} user records from {STATS_FILE}.')
    print(f"Developer ID set to: {DEVELOPER_ID or 'Not Set'}")
    print('Bot is ready and slash commands are synced.')
    print('------')

# --- HELP COMMAND ---
@tree.command(name="help", description="Shows how to use the bot.")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🤖 Valorant Predictor Bot Help",
        description="This bot uses machine learning to predict Valorant match outcomes.",
        color=discord.Color.purple()
    )
    embed.add_field(
        name="`/predict` Command",
        value=(
            "Use this to start a prediction.\n"
            "**Parameters:**\n"
            "• `team_a`: The first team.\n"
            "• `team_b`: The second team.\n"
            "• `best_of`: The match format (Bo1, Bo3, Bo5)."
        ),
        inline=False
    )
    # The /stats command is now hidden from the public help message
    embed.add_field(
        name="⭐ Model Tiers Explained",
        value=(
            "Models are grouped by complexity:\n\n"
            "**⭐ Standard:** Baseline models for fundamental predictions.\n\n"
            "**⭐⭐ Advanced:** More powerful models for improved accuracy.\n\n"
            "**⭐⭐⭐ Deep Learning:** The most complex models to find deep patterns in data."
        ),
        inline=False
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)


# --- STATS COMMAND (Now Developer-Only) ---
@tree.command(name="stats", description="View bot usage statistics (Developer only).")
@is_developer() # Use the new custom check
async def stats_command(interaction: discord.Interaction):
    """Displays usage statistics for the bot."""
    total_predictions = sum(data['count'] for data in user_stats.values())
    total_users = len(user_stats)
    sorted_users = sorted(user_stats.items(), key=lambda item: item[1]['count'], reverse=True)

    embed = discord.Embed(
        title="📊 Bot Usage Statistics",
        description=f"**Total Predictions:** {total_predictions}\n**Unique Users:** {total_users}",
        color=discord.Color.blue()
    )
    leaderboard = [
        f"`{i+1}.` <@{user_id}> (`{data['username']}`) - **{data['count']}** predictions"
        for i, (user_id, data) in enumerate(sorted_users[:10])
    ]
    if leaderboard:
        embed.add_field(name="🏆 Top Users", value="\n".join(leaderboard), inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True) # Ephemeral so only you see it

@stats_command.error
async def stats_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    # This will now catch the error from our custom `is_developer` check
    if isinstance(error, app_commands.CheckFailure):
        await interaction.response.send_message("This command is restricted to the bot developer.", ephemeral=True)
    else:
        # For any other errors, print them to the console
        print(f"An unhandled error occurred in the stats command: {error}")
        await interaction.response.send_message("An unexpected error occurred.", ephemeral=True)

# --- PREDICT COMMAND (No changes needed here) ---
@tree.command(name="predict", description="Predict the winner of a Valorant match.")
@app_commands.describe(
    team_a="Name of Team A",
    team_b="Name of Team B",
    best_of="The format of the match (Bo1, Bo3, or Bo5)"
)
@app_commands.choices(best_of=[
    app_commands.Choice(name="Best of 1", value="Bo1"),
    app_commands.Choice(name="Best of 3", value="Bo3"),
    app_commands.Choice(name="Best of 5", value="Bo5"),
])
async def predict_command(
        interaction: discord.Interaction,
        team_a: str,
        team_b: str,
        best_of: app_commands.Choice[str]
):
    await interaction.response.defer(thinking=True)
    # --- Update User Stats ---
    user_id = str(interaction.user.id)
    username = str(interaction.user)
    if user_id in user_stats:
        user_stats[user_id]['count'] += 1
        user_stats[user_id]['username'] = username
    else:
        user_stats[user_id] = {'username': username, 'count': 1}
    await save_user_stats()
    # ---
    best_of_input = best_of.value
    sorted_teams = sorted([team_a.lower().strip(), team_b.lower().strip()])
    cache_key = f"{sorted_teams[0]}-{sorted_teams[1]}-{best_of_input.lower()}"
    results = prediction_cache.get(cache_key)
    is_cached = results is not None
    if not is_cached:
        try:
            results = run_all_models(team_a, team_b, best_of_input)
            prediction_cache[cache_key] = results
        except Exception as e:
            print(f"An error occurred during model execution: {e}")
            await interaction.followup.send("❌ An unexpected error occurred while running the models.")
            return
    valid_probs = [res['prob_a_wins'] for res in results if res['prob_a_wins'] is not None]
    if not valid_probs:
        await interaction.followup.send(embed=discord.Embed(title=" Prediction Failed", description="None of the models could generate a prediction.", color=discord.Color.red()))
        return
    average_prob = sum(valid_probs) / len(valid_probs)
    winner = team_a if average_prob > 0.5 else team_b
    winner_prob = average_prob if winner == team_a else 1 - average_prob
    confidence = abs(average_prob - 0.5)
    if confidence > 0.20: color = discord.Color.green()
    elif confidence > 0.05: color = discord.Color.blue()
    else: color = discord.Color.gold()
    embed = discord.Embed(title="📈 Valorant Match Prediction", description=f"## ⚔️ **{team_a}** vs **{team_b}**", color=color)
    embed.add_field(name="🏆 Overall Prediction", value=(f"**Predicted Winner: `{winner}`**\nConfidence: `{winner_prob:.2%}`\nAvg. P({team_a} wins): `{average_prob:.2%}`"), inline=False)
    embed.add_field(name="📋 Match Info", value=f"**Format:** `{best_of_input}`", inline=False)
    tiered_data = {'⭐': [], '⭐⭐': [], '⭐⭐⭐': []}
    for res in results:
        prob_a = res['prob_a_wins']
        row = [res['model'], "Failed", "N/A"]
        if prob_a is not None:
            model_winner = team_a if prob_a >= 0.5 else team_b
            model_confidence = f"{prob_a if prob_a >= 0.5 else 1 - prob_a:.2%}"
            row = [res['model'], model_winner, model_confidence]
        tiered_data[res['tier']].append(row)
    headers = ["Model", "Winner", "Confidence"]
    for tier, title in {'⭐': "⭐ Standard Models", '⭐⭐': "⭐⭐ Advanced Models", '⭐⭐⭐': "⭐⭐⭐ Deep Learning Models"}.items():
        if tiered_data[tier]:
            table_string = tabulate(tiered_data[tier], headers=headers, tablefmt="github")
            embed.add_field(name=title, value=f"```\n{table_string}\n```", inline=False)
    footer_text = f"Aggregated from {len(valid_probs)}/{len(results)} successful models."
    if is_cached: footer_text += " (Result from cache ⚡)"
    embed.set_footer(text=footer_text)
    await interaction.followup.send(embed=embed)

# --- Run the Bot ---
client.run(TOKEN)
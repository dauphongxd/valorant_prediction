# bot.py (Updated with tiered tables and confidence column)

import os
import discord
from discord import app_commands
from dotenv import load_dotenv
import typing
from tabulate import tabulate

# --- Ensure the script's working directory is correct ---
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

# --- Import your existing ML pipeline function ---
from main import run_all_models

# --- Load Environment Variables ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
if not TOKEN:
    raise ValueError("DISCORD_TOKEN not found in .env file. Please create one.")

# --- Model "Smartness" Tiers ---
MODEL_TIERS = {
    'Logistic Regression': '⭐', 'Decision Tree': '⭐', 'Random Forest': '⭐',
    'CatBoost': '⭐⭐', 'LightGBM': '⭐⭐', 'XGBoost': '⭐⭐',
    'TabNet': '⭐⭐⭐', 'TabTransformer': '⭐⭐⭐',
}

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


@client.event
async def on_ready():
    await tree.sync()
    print(f'Logged in as {client.user} (ID: {client.user.id})')
    print('Bot is ready and slash commands are synced.')
    print('------')


# --- FULLY UPDATED AND CORRECTED BOT COMMAND ---

@tree.command(name="predict", description="Predict the winner of a Valorant match.")
@app_commands.describe(
    team_a="Name of Team A",
    team_b="Name of Team B",
    best_of="The format of the match (Bo1, Bo3, or Bo5)",
    map_name="The map name (optional, leave blank if not applicable)"
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
        best_of: app_commands.Choice[str],
        map_name: typing.Optional[str] = None
):
    """The slash command function for making predictions."""
    await interaction.response.defer(thinking=True)

    map_input = map_name or 'N/A'
    best_of_input = best_of.value

    try:
        # 1. Run your existing model pipeline
        print(f"Running prediction for: {team_a} vs {team_b} ({best_of_input} on {map_input})")
        results = run_all_models(team_a, team_b, map_input, best_of_input)

        # 2. Process results and calculate aggregate
        valid_probs = [res['prob_a_wins'] for res in results if res['prob_a_wins'] is not None]

        if not valid_probs:
            embed = discord.Embed(
                title=" Prediction Failed",
                description="None of the models were able to generate a prediction for this matchup.",
                color=discord.Color.red()
            )
            await interaction.followup.send(embed=embed)
            return

        # 3. Create the embed
        average_prob = sum(valid_probs) / len(valid_probs)
        winner = team_a if average_prob > 0.5 else team_b
        winner_prob = average_prob if winner == team_a else 1 - average_prob

        confidence = abs(average_prob - 0.5)
        if confidence > 0.20:
            color = discord.Color.green()
        elif confidence > 0.05:
            color = discord.Color.blue()
        else:
            color = discord.Color.gold()

        embed = discord.Embed(
            title="📈 Valorant Match Prediction",
            description=f"## ⚔️ **{team_a}** vs **{team_b}**",
            color=color
        )
        embed.add_field(name="🏆 Overall Prediction", value=(
            f"**Predicted Winner: `{winner}`**\n" f"Confidence: `{winner_prob:.2%}`\n" f"Avg. P({team_a} wins): `{average_prob:.2%}`\n" f"Avg. P({team_b} wins): `{1 - average_prob:.2%}`"),
                        inline=False)
        embed.add_field(name="📋 Match Info", value=(f"**Format:** `{best_of_input}`\n**Map:** `{map_input}`"),
                        inline=False)

        # --- FIX: DYNAMIC PADDING LOGIC ---
        # 1. Calculate the maximum length of any model name. Add 1 for a space.
        max_len = max(len(res['model']) for res in results) if results else 18  # Default to 18 if no results

        tiered_data = {'⭐': [], '⭐⭐': [], '⭐⭐⭐': []}

        # 2. Process results using the calculated max_len for padding
        for res in results:
            model_name = res['model']
            prob_a = res['prob_a_wins']
            tier = MODEL_TIERS.get(model_name, '⭐')

            if prob_a is None:
                row = [model_name, "Failed", "N/A"]
            else:
                model_winner = team_a if prob_a >= 0.5 else team_b
                model_confidence = f"{prob_a if prob_a >= 0.5 else 1 - prob_a:.2%}"
                row = [model_name, model_winner, model_confidence]

            tiered_data[tier].append(row)

        # 3. Create a separate, aligned field for each tier using the dynamic max_len
        tier_titles = {
            '⭐': "⭐ Baseline Models",
            '⭐⭐': "⭐⭐ Advanced Ensemble Models",
            '⭐⭐⭐': "⭐⭐⭐ Deep Learning Models"
        }

        headers = ["Model", "Winner", "Confidence"]

        for tier, title in tier_titles.items():
            if tiered_data[tier]:
                # This one line creates the entire perfectly-aligned table!
                table_string = tabulate(tiered_data[tier], headers=headers, tablefmt="github")

                # Wrap it in a Discord code block
                value_str = f"```\n{table_string}\n```"
                embed.add_field(name=title, value=value_str, inline=False)

        embed.set_footer(text=f"Aggregated from {len(valid_probs)}/{len(results)} successful models.")
        await interaction.followup.send(embed=embed)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        await interaction.followup.send(f"❌ An unexpected error occurred. Please check the console logs.\n`{e}`")


# --- Run the Bot ---
client.run(TOKEN)
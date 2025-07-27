# main.py (Updated to enforce valid team names)

import sys
import os
from importlib import import_module
from typing import Optional

# --- CONFIGURATION: Team Name Mapping ---
# Maps full team names to their common abbreviations. The system will prioritize the full name.
TEAM_NAME_MAPPING = {
    "100 Thieves": "100T",
    "2Game Esports": "2G",
    "All Gamers": "AG",
    "Apeks": "APEKS",
    "Attacking Soul Esports": "ASE",
    "BBL Esports": "BBL",
    "BLEED": "BLD",
    "BOOM Esports": "BOOM",
    "Bilibili Gaming": "BLG",
    "Cloud9": "C9",
    "DRX": "DRX",
    "DetonatioN FocusMe": "DFM",
    "Douyu Gaming": "DYG",
    "Dragon Ranger Gaming": "DRG",
    "EDward Gaming": "EDG",
    "Evil Geniuses": "EG",
    "FNATIC": "FNC",
    "FURIA": "FUR",
    "FUT Esports": "FUT",
    "Four Angry Men": "4AM",
    "FunPlus Phoenix": "FPX",
    "G2 Esports": "G2",
    "GIANTX": "GX",
    "Gank Gaming": "GNK",
    "Gen.G": "GENG",
    "Gentle Mates": "M8",
    "Giants Gaming": "GIA",
    "Global Esports": "GE",
    "Guangzhou Huadu Bilibili Gaming(Bilibili Gaming)": "BLG",
    "Invincible Gaming": "IG",
    "JD Mall JDG Esports(JDG Esports)": "JDG",
    "JDG Esports": "JDG",
    "KOI": "KOI",
    "KRU Esports": "KRU",
    "Karmine Corp": "KC",
    "Kingzone": "KZ",
    "LEVIATAN": "LEV",
    "LOUD": "LOUD",
    "MIBR": "MIBR",
    "Monarch Effect": "ME",
    "Movistar KOI(KOI)": "KOI",
    "NRG": "NRG",
    "Natus Vincere": "NAVI",
    "Night Wings Gaming": "NWG",
    "Nongshim RedForce": "NS",
    "Nova Esports": "NOVA",
    "Number One Player": "NOP",
    "Paper Rex": "PRX",
    "Rare Atom": "RA",
    "Rex Regum Qeon": "RRQ",
    "Royal Never Give Up": "RNG",
    "Sentinels": "SEN",
    "Shenzhen NTER": "NTER",
    "T1": "T1",
    "TALON": "TLN",
    "TYLOO": "TYL",
    "Team Bunny": "TBNY",
    "Team Heretics": "TH",
    "Team Liquid": "TL",
    "Team Secret": "TS",
    "Team SuperBusS": "TSB",
    "Team Vitality": "VIT",
    "Titan Esports Club": "TEC",
    "Totoro Gaming": "TRG",
    "Trace Esports": "TE",
    "VISA KRU(KRU Esports)": "KRU",
    "Weibo Gaming": "WBG",
    "Wolves Esports": "WLG",
    "Xi Lai Gaming": "XLG",
    "ZETA DIVISION": "ZETA",
}

# Create a reverse mapping for efficient lookups (abbreviation -> full name)
# All keys are converted to lowercase for case-insensitive matching
REVERSE_TEAM_MAPPING = {v.lower(): k for k, v in TEAM_NAME_MAPPING.items()}

# --- CONFIGURATION: Define all models and their properties ---
MODELS_CONFIG = {
    'CatBoost': {
        'path': 'models/catboost',
        'module_name': 'catboost_predict',
        'entry_function': 'get_prediction',
        'display_name': 'Advanced Predictor A',
        'tier': '⭐⭐'
    },
    'LightGBM': {
        'path': 'models/light_gbm',
        'module_name': 'lightgbm_predict',
        'entry_function': 'get_prediction',
        'display_name': 'Advanced Predictor B',
        'tier': '⭐⭐'
    },
    'XGBoost': {
        'path': 'models/xgboost',
        'module_name': 'xg_boost_predict',
        'entry_function': 'get_prediction',
        'display_name': 'Advanced Predictor C',
        'tier': '⭐⭐'
    },
    'Random Forest': {
        'path': 'models/random_forest',
        'module_name': 'random_forest_predict',
        'entry_function': 'get_prediction',
        'display_name': 'Standard Analyst A',
        'tier': '⭐'
    },
    'Decision Tree': {
        'path': 'models/decision_tree',
        'module_name': 'decisiontree_predict',
        'entry_function': 'get_prediction',
        'display_name': 'Standard Analyst B',
        'tier': '⭐'
    },
    'Logistic Regression': {
        'path': 'models/logistic_regression',
        'module_name': 'logistic_regression_predict',
        'entry_function': 'get_prediction',
        'display_name': 'Standard Analyst C',
        'tier': '⭐'
    },
    'TabNet': {
        'path': 'models/tabnet',
        'module_name': 'tabnet_predict',
        'entry_function': 'get_prediction',
        'display_name': 'Deep Learning Oracle A',
        'tier': '⭐⭐⭐'
    },
    'TabTransformer': {
        'path': 'models/tab_transformer',
        'module_name': 'tab_transformer_predict',
        'entry_function': 'get_prediction',
        'display_name': 'Deep Learning Oracle B',
        'tier': '⭐⭐⭐'
    }
}


# --- UPDATED: Team Name Normalization Function ---
def normalize_team_name(name: str) -> Optional[str]:
    """
    Converts a team abbreviation or alternate name to its official, full name.
    Returns None if no mapping is found, indicating an invalid team.
    """
    cleaned_name = name.strip().lower()
    # Check if the input is an abbreviation (e.g., 'c9')
    if cleaned_name in REVERSE_TEAM_MAPPING:
        return REVERSE_TEAM_MAPPING[cleaned_name]
    # Check if the input is already a full name (but maybe with wrong casing)
    for full_name, abbr in TEAM_NAME_MAPPING.items():
        if full_name.lower() == cleaned_name:
            return full_name
    # If no mapping found, return None
    return None


# --- MAIN PIPELINE LOGIC ---

def run_all_models(team_a, team_b, best_of):
    """
    Iterates through all configured models, runs their prediction, and collects the results.
    Map name is no longer a parameter here.
    """
    all_predictions = []
    map_name = 'N/A'  # Hardcode map_name since it's no longer a user input

    for model_name, config in MODELS_CONFIG.items():
        print(f"\n--- Running {model_name} ---")
        model_dir = config['path']

        result_payload = {
            'model': config['display_name'],
            'tier': config['tier'],
            'prob_a_wins': None
        }

        sys.path.insert(0, model_dir)
        try:
            predict_module = import_module(config['module_name'])
            predict_func = getattr(predict_module, config['entry_function'])

            # The underlying model functions may still expect map_name, so we pass the default.
            prob_a_wins = predict_func(
                team_a=team_a,
                team_b=team_b,
                map_name=map_name,  # Pass the hardcoded 'N/A' value
                best_of=best_of,
                artifacts_dir=model_dir
            )

            result_payload['prob_a_wins'] = prob_a_wins
            all_predictions.append(result_payload)

            if prob_a_wins is not None:
                print(f"  > Prediction complete: {team_a} win prob = {prob_a_wins:.2%}")
            else:
                print(f"  > Prediction failed.")

        except Exception as e:
            print(f"  > An unexpected error occurred while running {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_predictions.append(result_payload)
        finally:
            if model_dir in sys.path:
                sys.path.remove(model_dir)

    return all_predictions


def main():
    """The main Command Line Interface (CLI) for the prediction pipeline."""
    print("--- Valorant Match Prediction Aggregator ---")
    print("This tool will run predictions across all 8 available models.")

    while True:
        print("\n--- New Prediction ---")
        team_a_input = input("Enter name for Team A: ").strip()
        if team_a_input.lower() == 'exit': break

        team_b_input = input("Enter name for Team B: ").strip()
        if team_b_input.lower() == 'exit': break

        # --- Normalize and Validate Team Names ---
        team_a = normalize_team_name(team_a_input)
        team_b = normalize_team_name(team_b_input)

        if team_a is None:
            print(f"\n[Error] Invalid team name: '{team_a_input}'. No data has been trained on this team.")
            continue  # Restart the loop

        if team_b is None:
            print(f"\n[Error] Invalid team name: '{team_b_input}'. No data has been trained on this team.")
            continue  # Restart the loop

        print(f" > Normalized names: '{team_a_input}' -> '{team_a}', '{team_b_input}' -> '{team_b}'")

        while True:
            best_of_num = input("Enter Best Of (1, 3, or 5): ").strip()
            if best_of_num in ['1', '3', '5']:
                best_of = f"Bo{best_of_num}"
                break
            else:
                print("Invalid input. Please enter 1, 3, or 5.")

        # Run the entire pipeline with normalized names
        results = run_all_models(team_a, team_b, best_of)

        # --- Aggregate and Display Final Results ---
        print("\n" + "=" * 50)
        print("          FINAL PREDICTION SUMMARY")
        print("=" * 50)
        print(f"Matchup: {team_a} vs. {team_b}")
        print(f"Format: {best_of}")
        print("-" * 50)

        valid_probs = []
        for result in results:
            prob = result['prob_a_wins']
            if prob is not None:
                print(f"  - {result['model']:<25}: {team_a} wins @ {prob:.2%}")
                valid_probs.append(prob)
            else:
                print(f"  - {result['model']:<25}: Prediction Failed")

        print("-" * 50)

        if not valid_probs:
            print("No models were able to generate a prediction.")
        else:
            average_prob = sum(valid_probs) / len(valid_probs)
            winner = team_a if average_prob > 0.5 else team_b

            print("AGGREGATED RESULT (AVERAGE OF VALID MODELS):")
            print(f"  > Predicted Winner:      {winner}")
            print(f"  > Avg P({team_a} wins): {average_prob:.2%}")
            print(f"  > Avg P({team_b} wins): {(1 - average_prob):.2%}")

        print("=" * 50)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    main()
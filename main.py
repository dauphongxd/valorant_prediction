# main.py (Updated to remove map_name)

import sys
import os
from importlib import import_module

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


# --- MAIN PIPELINE LOGIC ---

def run_all_models(team_a, team_b, best_of):
    """
    Iterates through all configured models, runs their prediction, and collects the results.
    Map name is no longer a parameter here.
    """
    all_predictions = []
    map_name = 'N/A' # Hardcode map_name since it's no longer a user input

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
                map_name=map_name, # Pass the hardcoded 'N/A' value
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
        team_a = input("Enter name for Team A: ").strip()
        if team_a.lower() == 'exit': break

        team_b = input("Enter name for Team B: ").strip()
        if team_b.lower() == 'exit': break

        while True:
            best_of_num = input("Enter Best Of (1, 3, or 5): ").strip()
            if best_of_num in ['1', '3', '5']:
                best_of = f"Bo{best_of_num}"
                break
            else:
                print("Invalid input. Please enter 1, 3, or 5.")

        # Run the entire pipeline without map_name
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
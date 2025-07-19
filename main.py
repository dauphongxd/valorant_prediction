import sys
import os
from importlib import import_module

# --- CONFIGURATION: Define all models and their properties ---
# This is now simpler. We just define the path and the new, unified entry point function.
MODELS_CONFIG = {
    'CatBoost': {
        'path': 'models/catboost',
        'module_name': 'catboost_predict',
        'entry_function': 'get_prediction',
    },
    'LightGBM': {
        'path': 'models/light_gbm',
        'module_name': 'lightgbm_predict',
        'entry_function': 'get_prediction',
    },
    'XGBoost': {
        'path': 'models/xgboost',
        'module_name': 'xg_boost_predict',
        'entry_function': 'get_prediction',
    },
    'Random Forest': {
        'path': 'models/random_forest',
        'module_name': 'random_forest_predict',
        'entry_function': 'get_prediction',
    },
    'Decision Tree': {
        'path': 'models/decision_tree',
        'module_name': 'decisiontree_predict',
        'entry_function': 'get_prediction',
    },
    'Logistic Regression': {
        'path': 'models/logistic_regression',
        'module_name': 'logistic_regression_predict',
        'entry_function': 'get_prediction',
    },
    'TabNet': {
        'path': 'models/tabnet',
        'module_name': 'tabnet_predict',
        'entry_function': 'get_prediction',
    },
    'TabTransformer': {
        'path': 'models/tab_transformer',
        'module_name': 'tab_transformer_predict',
        'entry_function': 'get_prediction',
    }
}


# --- MAIN PIPELINE LOGIC ---

def run_all_models(team_a, team_b, map_name, best_of):
    """
    Iterates through all configured models, runs their prediction, and collects the results.
    """
    all_predictions = []

    for model_name, config in MODELS_CONFIG.items():
        print(f"\n--- Running {model_name} ---")
        model_dir = config['path']

        # Temporarily add the model's directory to the Python path for importing
        sys.path.insert(0, model_dir)
        try:
            # Dynamically import the prediction module
            predict_module = import_module(config['module_name'])

            # Get the unified entry point function from the module
            predict_func = getattr(predict_module, config['entry_function'])

            # Call the function, passing the artifacts directory
            # The function inside the module is now responsible for its own logic
            prob_a_wins = predict_func(
                team_a=team_a,
                team_b=team_b,
                map_name=map_name,
                best_of=best_of,
                artifacts_dir=model_dir  # This is the crucial change
            )

            all_predictions.append({'model': model_name, 'prob_a_wins': prob_a_wins})
            if prob_a_wins is not None:
                print(f"  > Prediction complete: {team_a} win prob = {prob_a_wins:.2%}")
            else:
                print(f"  > Prediction failed.")

        except Exception as e:
            print(f"  > An unexpected error occurred while running {model_name}: {e}")
            import traceback
            traceback.print_exc()  # Print full error for better debugging
            all_predictions.append({'model': model_name, 'prob_a_wins': None})
        finally:
            # IMPORTANT: Clean up the path
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

        map_name_input = input("Enter Map Name (or leave blank for models that can handle it): ").strip()
        map_name = map_name_input or 'N/A'

        while True:
            best_of_num = input("Enter Best Of (1, 3, or 5): ").strip()
            if best_of_num in ['1', '3', '5']:
                best_of = f"Bo{best_of_num}"
                break
            else:
                print("Invalid input. Please enter 1, 3, or 5.")

        # Run the entire pipeline
        results = run_all_models(team_a, team_b, map_name, best_of)

        # --- Aggregate and Display Final Results ---
        print("\n" + "=" * 50)
        print("          FINAL PREDICTION SUMMARY")
        print("=" * 50)
        print(f"Matchup: {team_a} vs. {team_b}")
        print(f"Format: {best_of} on {map_name if map_name_input else 'Not Specified'}")
        print("-" * 50)

        valid_probs = []
        for result in results:
            prob = result['prob_a_wins']
            if prob is not None:
                print(f"  - {result['model']:<20}: {team_a} wins @ {prob:.2%}")
                valid_probs.append(prob)
            else:
                print(f"  - {result['model']:<20}: Prediction Failed")

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
import json
import os
import logging


class Translator:
    def __init__(self, locales_dir='locales'):
        """
        Initializes the translator by loading all .json language files
        from the specified directory.
        """
        self.locales = {}
        self.default_lang = 'en'  # Set English as the fallback language

        if not os.path.exists(locales_dir):
            logging.error(f"Locales directory '{locales_dir}' not found.")
            return

        for filename in os.listdir(locales_dir):
            if filename.endswith('.json'):
                # Extract the language code from the filename (e.g., 'en' from 'en.json')
                lang_code = filename.split('.')[0]
                filepath = os.path.join(locales_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.locales[lang_code] = json.load(f)
                        logging.info(f"Successfully loaded language file: {filename}")
                except Exception as e:
                    logging.error(f"Failed to load or parse {filename}: {e}")

    def get_string(self, key_path: str, lang: str, **kwargs):
        """
        Gets a translated string using its key.

        Args:
            key_path (str): The dot-separated key (e.g., "help_command.title").
            lang (str): The user's language code from Discord (e.g., "en-US", "vi").
            **kwargs: Placeholder values to format into the string.

        Returns:
            str: The translated and formatted string.
        """
        # Get the base language code (e.g., 'vi' from 'vi-VN')
        base_lang = lang.split('-')[0]

        # Find the text, falling back to the default language if needed
        keys = key_path.split('.')

        # Try to get from the requested language
        primary_dict = self.locales.get(base_lang, {})
        # Fallback dictionary is always English
        fallback_dict = self.locales.get(self.default_lang, {})

        # Traverse the dictionaries to find the string
        text_template = primary_dict
        for key in keys:
            text_template = text_template.get(key)
            if text_template is None:
                break

        if text_template is None:
            text_template = fallback_dict
            for key in keys:
                text_template = text_template.get(key)
                if text_template is None:
                    break

        if not isinstance(text_template, str):
            # If the key is not found or is not a string, return the key path itself as an error indicator
            logging.warning(f"Translation key '{key_path}' not found for language '{base_lang}' or fallback 'en'.")
            return key_path

        # Replace placeholders like {username} with actual values
        try:
            return text_template.format(**kwargs)
        except KeyError as e:
            logging.error(f"Missing placeholder value for key '{key_path}': {e}")
            return text_template  # Return the unformatted template on error


# Create a single, global instance that the bot will import and use
translator = Translator()
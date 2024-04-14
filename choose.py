"""Choose model to build"""
import json
import sys
import os
from typing import Dict, List

def update_model_settings(model_type: str) -> None:
    """
    Updates the model settings file based on the specified model type.

    Args:
        model_type (str): The type of model configuration to apply. Valid options are 'solo' or 'two'.

    Raises:
        ValueError: If an invalid model type is specified.
        FileNotFoundError: If the settings file does not exist and cannot be removed.
    """
    settings_path = "model-config.json"

    # Define basic structure of model configuration data
    configuration: Dict[str, any] = {
        "name": "uplift-predictor",
        "implementation": "inference.UpliftModel",
        "parameters": {"uri": ""}
    }

    # Assign the correct model URI based on the command line argument
    if model_type == "default":
        configuration["parameters"]["uri"] = "one_model.joblib"
    elif model_type == "alt":
        configuration["parameters"]["uri"] = "two_model.joblib"
    else:
        raise ValueError("Invalid model type specified. Choose either 'default' or 'alt'.")

    # Attempt to remove the existing settings file if it exists
    try:
        os.remove(settings_path)
    except FileNotFoundError:
        print(f"Notice: The file {settings_path} was not found and will be created.")

    # Write the updated configuration to the file
    with open(settings_path, "w") as file:
        json.dump(configuration, file, indent=4)
    print(f"Model settings updated successfully in {settings_path}.")

def main() -> None:
    """
    Main function that processes command line arguments to update model settings.
    """
    if len(sys.argv) != 2 or sys.argv[1] not in ["default", "alt"]:
        print("Usage error: The script takes exactly 1 argument: {default, alt}")
        sys.exit(-1)

    # Update model settings based on the provided argument
    update_model_settings(sys.argv[1])

if __name__ == "__main__":
    main()

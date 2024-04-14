import json
import numpy as np
import requests
import pytest

def make_inference_payload() -> str:
    """
    Creates a JSON string payload for sending an inference request. Data has been
    uniquely modified to ensure the uniqueness of the test inputs.

    Returns:
        str: A JSON string of the request payload.
    """
    inputs = {
        "data": [
            [58, "M", 1499201232, 1504293512.0, 5092280.0],
            [53, "F", 1522430280, 1531511552.0, 9081272.0], 
            [44, "M", 1507409602, 1534023038.0, 26613436.0], 
            [115, "F", 1530235581, 1550262677.0, 20026096.0],
        ]
    }
    return json.dumps(inputs)

def construct_inference_request(input_string: str) -> dict:
    """
    Constructs the inference request dictionary using the provided input string.

    Args:
        input_string (str): The JSON string containing inference input data.

    Returns:
        dict: A dictionary structured as an inference request.
    """
    return {
        "inputs": [
            {
                "name": "predict_request",
                "shape": [len(input_string)],
                "datatype": "BYTES",
                "data": [input_string],
            }
        ]
    }

@pytest.mark.parametrize("endpoint", ["http://localhost:9000/v2/models/uplift-predictor/infer"])
def test_inference_response(endpoint: str):
    """
    Ensures the model server's inference endpoint responds with a 200 OK status,
    indicating successful communication and response generation.

    Args:
        endpoint (str): The URL of the inference API endpoint.
    """
    inputs_string = make_inference_payload()
    inference_request = construct_inference_request(inputs_string)
    response = requests.post(endpoint, json=inference_request)

    assert response.status_code == 200, "Expected a 200 OK response from the model server."

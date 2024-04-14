from mlserver import MLModel, types
from mlserver.utils import get_model_uri
from mlserver.codecs import StringCodec
from joblib import load
from typing import Dict, Any
import numpy as np
import json


class UpliftModel(MLModel):
    """
    An asynchronous ML server model for predicting customer uplift using an uplift model.
    """
    async def initialize_model(self) -> bool:
        """
        Asynchronously loads the model from a URI specified in the settings.
        
        Returns:
            bool: True if the model is successfully loaded and ready, False otherwise.
        """
        model_path = await get_model_uri(self._settings)
        self.uplift_estimator = load(model_path)
        self.is_ready = True
        return self.is_ready

    async def perform_prediction(self, request: types.InferenceRequest) -> types.InferenceResponse:
        """
        Process an inference request and returns the prediction results.
        
        Args:
            request (types.InferenceRequest): The inference request object containing input data.
        
        Returns:
            types.InferenceResponse: The response object containing the prediction results.
        """
        try:
            decoded_request = self._parse_request(request).get("predict_request", {})
            feature_array = np.array(decoded_request.get("data", []))

            prediction_result = {"success": True, "prediction": self.uplift_estimator.predict(feature_array)}

        except Exception as e:
            prediction_result = {"success": False, "prediction": None}

        response_payload = json.dumps(prediction_result.__repr__()).encode("UTF-8")

        return types.InferenceResponse(
            id=request.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                types.ResponseOutput(
                    name="prediction_output",
                    shape=[len(response_payload)],
                    datatype="BYTES",
                    data=[response_payload],
                    parameters=types.Parameters(content_type="application/json"),
                )
            ],
        )

    def _parse_request(self, request: types.InferenceRequest) -> Dict[str, Any]:
        """
        Decodes and extracts JSON data from an inference request's inputs.
        
        Args:
            request (types.InferenceRequest): The request from which to extract data.
        
        Returns:
            Dict[str, Any]: A dictionary containing the decoded data.
        """
        decoded_inputs = {}
        for input_data in request.inputs:
            decoded_inputs[input_data.name] = json.loads(
                "".join(self.decode(input_data, default_codec=StringCodec))
            )

        return decoded_inputs

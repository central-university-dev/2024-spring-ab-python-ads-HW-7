import logging
from mlserver import MLModel, types
from mlserver.utils import get_model_uri
from mlserver.codecs import StringCodec
from joblib import load
from typing import Dict, Any
import json

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class IrisPredictor(MLModel):
    async def load(self) -> bool:
        model_uri = await get_model_uri(self._settings)

        self.model = load(model_uri)
        self.ready = True
        return self.ready

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        try:
            request = self._extract_json(payload).get("echo_request", {})
            flowers = request.get("flowers", [])

            output_data = {"success": True}
            output_data["prediction"] = self.model.predict(flowers)

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            output_data = {
                "success": "False",
                "prediction": "NULL",
            }

        response_bytes = json.dumps(output_data.__repr__()).encode("UTF-8")

        return types.InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_verison=self.version,
            outputs=[
                types.ResponseOutput(
                    name="echo_response",
                    shape=[len(response_bytes)],
                    datatype="BYTES",
                    data=[response_bytes],
                    parameters=types.Parameters(content_type="str")
                )
            ]
        )

    def _extract_json(self, payload: types.InferenceRequest) -> Dict[str, Any]:
        inputs = {}
        for inp in payload.inputs:
            inputs[inp.name] = json.loads(
                "".join(self.decode(inp, default_codec=StringCodec))
            )

        return inputs

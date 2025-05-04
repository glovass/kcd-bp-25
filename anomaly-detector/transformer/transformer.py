#!/usr/bin/env python3
import argparse
from kserve import ModelServer, model_server, InferInput, InferOutput, InferRequest, InferResponse
from kserve.model import PredictorConfig, PredictorProtocol
from typing import Dict, Union

import kserve
import numpy as np
import joblib
import re


class LogTransformer(kserve.Model):
    def __init__(
        self,
        name: str,
        predictor_config: PredictorConfig,
    ):
        super().__init__(
            name,
            predictor_config,
            return_response_headers=True,
        )
        self.ready = True

    def __extract_features(self, logs):
        features = []
        for log in logs:
            log_level = 0  # 0=INFO, 1=ERROR, 2=WARNING, 3=CRITICAL
            if "ERROR" in log:
                log_level = 1
            elif "WARNING" in log:
                log_level = 2
            elif "CRITICAL" in log:
                log_level = 3

            has_error_keyword = int("error" in log.lower())
            has_success = int("success" in log.lower())
            features.append([log_level, has_error_keyword, has_success])
        return np.array(features)

    def __strip_timestamp(self, text):
        return re.sub(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ', '', text)

    def preprocess(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferRequest]:
        logs = payload.inputs[0].data
        print(logs)
        data = [self.__strip_timestamp(log) for log in logs]
        data = self.__extract_features(data)

        scaler = joblib.load('scaler.pkl')
        scaled_data = scaler.transform(data)
        print(scaled_data)
        print(list(scaled_data.shape))

        infer_inputs = [
            InferInput(
                name="INPUT__0",
                datatype="FP32",
                shape=list(scaled_data.shape),
                data=scaled_data.tolist(),
            )
        ]
        print(infer_inputs)

        return InferRequest(model_name=self.name, infer_inputs=infer_inputs)

    def postprocess(
        self,
        infer_response: Union[Dict, InferResponse],
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None,
    ) -> Union[Dict, InferResponse]:
        response = infer_response.outputs[0].data
        decoded_response = [ "Normal" if pred == 1 else "Anomaly" for pred in response]
        decoded_infer_output = InferOutput(
                name="OUTPUT_0",
                datatype="BYTES",
                shape=infer_response.outputs[0].shape,
                data=decoded_response
            )
        return InferResponse(infer_response.id, infer_response.model_name, [decoded_infer_output])


parser = argparse.ArgumentParser(parents=[model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = LogTransformer(
        args.model_name,
        PredictorConfig(
            args.predictor_host,
            args.predictor_protocol,
            args.predictor_use_ssl,
            args.predictor_request_timeout_seconds,
            args.predictor_request_retries,
            args.enable_predictor_health_check,
        ),
    )
    ModelServer().start([model])

import pdb
import grpc
import logging
import numpy as np

from openai import OpenAI


class BaseModuleVLLM(object):
    def __init__(self, common_config, model_config):
        self.model_config = model_config
        self.common_config = common_config
        openai_api_key = self.common_config['api_key']
        openai_api_base = "http://" + str(self.common_config['ip']) + ":" + str(self.common_config['port']) + "/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        
    def add_metadata(self, metadata, num_request, batch_size):
        metadata[self.__class__.__name__]['num_request'] += num_request
        metadata[self.__class__.__name__]['total_batch_size'] += batch_size
        return metadata
        
        
    def predict(self, request_id, info, metadata, *args, **kwargs):
        pass


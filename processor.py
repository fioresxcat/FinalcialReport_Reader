from page_align import PageAligner
from table_detection import TableDetector
from text_detection import BCTCTextDetector
from ocr import BCTCOCR
from splitter import BCTCSplitter
from information_extraction.information_extraction import BCTCTableInformationExtraction
from post_processor import BCTCPostProcessor
from inpout import Input, Output


class BCTCProcessor:
    def __init__(self, common_config, model_config):
        self.common_config = common_config
        self.model_config = model_config
        self.modules = [
            PageAligner(common_config['triton_server'], model_config),
            TableDetector(common_config['triton_server'], model_config['triton_models']['table_detection']),
            BCTCTextDetector(common_config['triton_server'], model_config['triton_models']['text_detection']),
            BCTCOCR(common_config['triton_server'], model_config['triton_models']['ocr']),
            BCTCSplitter(common_config['vllm_server'], model_config),
            BCTCTableInformationExtraction(common_config, model_config),
            BCTCPostProcessor(common_config['vllm_server'], model_config)
        ]


    def predict(self, request_id, inp):
        metadata = {}
        for module in self.modules:
            out = Output()
            if module.__class__.__name__ not in metadata.keys():
                metadata[module.__class__.__name__] = {
                    'num_request': 0,
                    'total_batch_size': 0
                }
            out, metadata = module.predict(request_id, inp, out, metadata)
            # end whole process if 1 module gets error
            if out.error_code == 0:
                inp.set_data(out.get_data())
            else:
                break
        return out, metadata


if __name__ == '__main__':
    pass
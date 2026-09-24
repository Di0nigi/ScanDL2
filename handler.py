import torch

from ScanDL2 import ScanDL2

class EndpointHandler:
    def __init__(self, path: str = ""):
       
        self.models = {
            "sentence":
            ScanDL2(
            text_type='sentence',  
            bsz=2,
            save=None,  
            filename=None,
        ),
        "paragraph":
        ScanDL2(
                text_type='paragraph', 
                bsz=2,
                save=None,  
                filename=None,
                )
            }
        
        for m in self.models.values():
            # m.to(self.device)
            m.eval()

    def __call__(self, data):
        

        inputs = data.get("inputs", data)  

        parameters = data.get("parameters", {})

        text_type = parameters.get("text_type", "sentence")
        model = self.models[text_type]
        bsz = parameters.get("bsz", 2)

        if model.scandl_module.args.batch_size != bsz:
            model.scandl_module.args.batch_size = bsz
            model.fixdur_module.bsz = bsz
            model.fixdur_module.args["bsz"] = bsz
        
        
        if isinstance(inputs, str):
            texts = [inputs]
        elif isinstance(inputs, list):
            texts = inputs
        else:
            raise ValueError("'inputs' must be a string or list of strings.")

        
        with torch.no_grad():
            output = model(texts=texts)

        
        return output
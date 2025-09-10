import torch
from pytorch_lightning import LightningModule

# placeholder interpolation model
class SRModelPL(LightningModule): 
    def __init__(self):
        super(SRModelPL, self).__init__()
    def forward(self, x, custom_steps=100):
        sr = torch.nn.functional.interpolate(x, size=(512, 512), mode='nearest')
        return sr
    def predict_step(self,x):
        return self.forward(x)
    
    
def preprocess_model(model):
    if isinstance(model, LightningModule):
        print("Model is a LightningModule.")
        model.eval()
        return model
    elif isinstance(model, torch.nn.Module):
        print("Model is a torch.nn.Module. Wrapping in LightningModule.")
        class WrappedModel(LightningModule):
            def __init__(self, model):
                super(WrappedModel, self).__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)
            def predict(self, x):
                return self.forward(x)
        wrapped_model = WrappedModel(model)
        wrapped_model.eval()
        return wrapped_model
    elif model is None:
        print("No model provided. Using placeholder interpolation model.")
        placeholder_model = SRModelPL()
        placeholder_model.eval()
        return placeholder_model
    else:
        raise NotImplementedError("Model must be a LightningModule, torch.nn.Module, or 'None'-Placeholder.")

import numpy as np
import os

def analytic_predict(rpm_window: np.ndarray):
    r = rpm_window.squeeze()
    thrust = 1e-6 * (r**2)
    torque = 5e-8 * (r**2)
    return float(thrust.mean()), float(torque.mean())

class PropellerModel:
    def __init__(self, model_path=None, device=None):
        self.model = None
        self.device = device
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            try:
                import torch
                from propeller_lstm import StackedLSTM
                self.model = StackedLSTM()
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.model.eval()
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
            except Exception:
                self.model = None

    def predict(self, rpm_window: np.ndarray):
        if self.model is not None:
            import torch
            x = rpm_window.reshape(1, rpm_window.shape[0], 1).astype("float32")
            x = torch.from_numpy(x).to(self.device)
            with torch.no_grad():
                y = self.model(x).cpu().numpy().squeeze()
            return float(y[0]), float(y[1])
        return analytic_predict(rpm_window)

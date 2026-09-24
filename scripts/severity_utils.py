# scripts/severity_utils.py
def severity_from_ur(ur: float) -> int:
    """
    Map unbalance ratio (Ur) to severity class.
    0: Normal, 1: Mild, 2: Moderate, 3: Severe
    Adjust thresholds if paper specifies different values.
    """
    if ur is None:
        return 0
    if ur < 0.15:
        return 0
    if ur < 0.30:
        return 1
    if ur < 0.60:
        return 2
    return 3

def severity_from_thrust_loss(thrust_loss: float) -> int:
    if thrust_loss is None:
        return 0
    if thrust_loss < 0.05:
        return 0
    if thrust_loss < 0.15:
        return 1
    if thrust_loss < 0.35:
        return 2
    return 3

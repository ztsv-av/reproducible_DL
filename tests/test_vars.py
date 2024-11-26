from utils.vars import SEED, DEVICE

def test_seed():
    """
    Tests whether the seed in utils/vars.py equals to the expected one.
    """
    # expected values
    expected_seed = 1
    # check that the seeds are equal
    assert SEED == expected_seed, f"Seed defined in utils/vars.py changed! Expected seed is {expected_seed}."

def test_device():
    """
    Tests that the device used is CPU.
    """
    # expected device
    expected_device = "cpu"
    assert DEVICE.type == expected_device, f"Device is set to {DEVICE}, expected {expected_device}."

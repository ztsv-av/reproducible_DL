from utils.vars import SEED, DEVICE, EPOCHS, LR

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

def test_epochs():
    """
    Tests that the number of epochs does not exceed 5.
    """
    # expected number of epochs
    expected_epochs = 2
    assert EPOCHS == expected_epochs, f"Number of epochs is set to {EPOCHS}, expected {expected_epochs}."

def test_lr():
    """
    Tests that the learning rate used during training is set to 0.01.
    """
    # expected maximum number of epochs
    expected_lr = 0.01
    assert LR == expected_lr, f"Learning rate is set to {LR}, expected {expected_lr}."

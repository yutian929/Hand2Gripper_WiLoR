from hand2gripper_wilor import HandDetector, WiLoRModel, HandRenderer
import os

def test_hand_detector():
    detector = HandDetector()
    assert detector is not None

def test_wi_model():
    model = WiLoRModel()
    assert model is not None

def test_hand_renderer():
    wilor_model = WiLoRModel()
    renderer = HandRenderer(model_cfg=wilor_model.model_cfg, faces=wilor_model.model.mano.faces)
    assert renderer is not None
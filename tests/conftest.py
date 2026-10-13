import base64
import copy
import math
import struct
from importlib.resources import files
import pytest
from vital2serum.serum import SerumPreset
from vital2serum.vital import VitalPreset


def wave(frame=None):
    frame=frame or [math.sin(2*math.pi*i/2048) for i in range(2048)]
    return dict(name='Generated sine',full_normalize=False,remove_all_dc=False,
                groups=[{'components':[{'type':'Wave Source','interpolation':0,'interpolation_style':1,
                  'keyframes':[{'position':0,'wave_data':base64.b64encode(struct.pack('<2048f',*frame)).decode()}]}]}])


def document():
    return dict(synth_version='1.5.5',preset_name='Generated Test',author='vital2serum test',
                comments='Synthetic test input, not a factory patch',preset_style='Test',
                macro1='Colour',settings={'wavetables':[wave(),None,None],
                 'lfos':[], 'modulations':[], 'sample':{'length':0},
                 'osc_1_on':1,'osc_2_on':0,'osc_3_on':0})

@pytest.fixture
def template():
    return SerumPreset.from_bytes(files('vital2serum').joinpath('data/Init-2.0.24.SerumPreset').read_bytes())

@pytest.fixture
def doc(): return document()

@pytest.fixture
def vital(doc): return VitalPreset(copy.deepcopy(doc))

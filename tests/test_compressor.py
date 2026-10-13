"""Compressor mapping pinned to measurements of Vital 1.0.7 and Serum 2.1.5."""
import math
import pytest
from vital2serum import compressor as c
from vital2serum.defaults import EFFECT_DEFAULTS


def values(**over):
    v={k:float(EFFECT_DEFAULTS['compressor'][k]) for k in c.KEYS}
    v.update(over)
    return v


def params(v,**kw):
    (name,type_id,p),=c.convert(v,**kw)[0]
    assert (name,type_id)==('FXComp',5)
    return p


def kinds(v,**kw):
    return [(kind,feature) for kind,feature,_ in c.convert(v,**kw)[1]]


def serum(**over):
    p=dict(kParamThresh=0.5,kParamRatio=1e6,kParamRatioBelow=1.0)
    for i in range(3): p.update({f'kParamThreshUD{i}':100.0,f'kParamDeadband{i}':10.0,f'kParamRatio{i}':1.0,f'kParamRatioBelow{i}':1.0,f'kParamGain{i}':0.0})
    p.update(over)
    return p


# --- Serum: what was measured --------------------------------------------------------------------
# Level (dB of the mean square, Vital's internal units) of a sine at the upper threshold of the
# low / mid / high band, the sine well inside the band, ThreshUD 100.
SERUM_KNEES = {0.3:(-4.89,-3.57,-5.39),0.4:(-14.36,-12.03,-15.33),0.5:(-26.52,-22.90,-28.11),
               0.6:(-41.40,-36.19,-43.73),0.7:(-58.97,-51.90,-62.19),0.8:(-79.25,-70.02,-83.49)}


def test_serum_threshold_follows_the_square_of_the_knob():
    for thresh,row in SERUM_KNEES.items():
        for band,measured in enumerate(row):
            assert c.serum_knee(thresh,100.0,band)==pytest.approx(measured,abs=0.12)


def test_serum_threshud_moves_one_band_and_scales_with_the_knob():
    for thresh,ud,measured in ((0.4,50,-19.71),(0.4,150,-4.35),(0.7,0,-98.92),(0.7,50,-75.41),(0.7,150,-28.38),(0.7,200,-4.86)):
        assert c.serum_knee(thresh,ud,1)==pytest.approx(measured,abs=0.05)
    assert c.serum_knee(0.5,-50,1)==c.serum_knee(0.5,0,1) and c.serum_knee(0.5,250,1)==c.serum_knee(0.5,200,1)


def test_serum_ratios():
    # A tone 24.2 dB over the mid band's threshold (Thresh 0.6): gain against Ratio and Ratio1.
    for ratio,band_ratio,measured in ((2.0,0.25,-3.02),(2.0,0.5,-6.03),(2.0,1.0,-12.04),(1e6,0.25,-6.03),(1e6,0.5,-12.04),(1e6,0.75,-17.99)):
        p=serum(kParamThresh=0.6,kParamRatio=ratio,kParamRatio1=band_ratio)
        assert c.serum_gain_db(-12.01,p,1)==pytest.approx(measured,abs=0.2)
    # A tone 33.1 dB under the lower threshold (Thresh 0.5, deadband 10): 0.5 is neutral, less expands.
    for below,band_below,measured in ((1.0,0.0,-33.11),(1.0,0.25,-16.57),(1.0,0.5,-0.03),(1.0,0.75,16.51),(1.0,1.0,33.05),(0.5,1.0,16.51),(0.5,0.6,3.28)):
        p=serum(kParamRatio=1.0,kParamRatioBelow=below,kParamRatioBelow1=band_below)
        assert c.serum_gain_db(-66.01,p,1)==pytest.approx(measured,abs=0.1)
    assert c.serum_gain_db(-120.0,serum(kParamRatio=1.0),1)==c.SERUM_MAX_BOOST_DB        # measured 35.97


def test_serum_deadband_is_kept_between_3_and_24_db():
    # Lower threshold of the mid band at Thresh 0.5 (upper one at -22.90) against Deadband1.
    for deadband,measured in ((0.0,-25.90),(3.0,-25.90),(7.0,-29.90),(11.6,-34.51),(20.0,-42.91),(24.0,-46.91),(60.0,-46.91)):
        p=serum(kParamRatio=1.0,kParamDeadband1=deadband)
        assert c.serum_gain_db(measured-10.0,p,1)==pytest.approx(10.0,abs=0.05) and c.serum_gain_db(measured+1.0,p,1)==0.0


def test_serum_makeup_doubles_at_limiter_ratios():
    for ratio,stored,measured in ((1.0,2.0,6.16),(4.0,4.0,12.25),(10.0,16.0,24.35),(10.0,31.0,30.10),(100.0,2.0,9.82),(1e6,4.0,17.26),(1e6,31.0,36.12)):
        assert c.serum_gain_db(-40.0,serum(kParamRatio=ratio,kParamRatioBelow=0.0,kParamMakeup=stored),1)==pytest.approx(measured,abs=0.02)


def test_serum_release_rate():
    # The limiter's gain came back by 4.3 dB per 100 ms (mid) and 9.2 dB (high) at kParamRelease 200.
    rate=20.0*math.log10(math.e)/200.0*100.0
    assert rate/c.SERUM_RELEASE_SCALE[1]==pytest.approx(4.3,abs=0.1) and rate/c.SERUM_RELEASE_SCALE[2]==pytest.approx(9.2,abs=0.1)


# --- Vital: what was measured --------------------------------------------------------------------
def through_vital(level,**band):
    """Gain of the whole compressor for a 262 Hz sine: 4.24 % of it lies in the untouched low band."""
    share=1.0/(1.0+(120.0/261.63)**4)
    gain=c.vital_gain_db(level+20.0*math.log10(share),**band)
    return 20.0*math.log10(share*10.0**(gain/20.0)+1.0-share)


def test_vital_static_curve():
    # Attack and release times made equal, so the detector reads the mean square itself.
    off=dict(upper_threshold=0.0,upper_ratio=0.0,lower_threshold=-80.0,lower_ratio=0.0)
    for level,band,measured in ((-9.0,dict(off,upper_threshold=-20.0,upper_ratio=1.0),-9.78),(-3.0,dict(off,upper_threshold=-20.0,upper_ratio=1.0),-14.72),
                                (-12.0,dict(off,upper_threshold=-20.0,upper_ratio=0.5),-3.61),(-60.0,dict(off,lower_threshold=-40.0,lower_ratio=0.8),15.98),
                                (-90.0,dict(off,lower_threshold=-40.0,lower_ratio=0.8),29.74),(-45.0,dict(off,lower_threshold=-40.0,lower_ratio=-1.0),-5.07),
                                (-30.0,dict(off,gain=10.0),9.74)):
        assert through_vital(level,**band)==pytest.approx(measured,abs=0.03)
    default=dict(upper_threshold=-25.0,upper_ratio=0.857,lower_threshold=-36.0,lower_ratio=0.8,gain=11.7)
    for level,measured in ((-39.0,14.10),(-30.0,11.42),(-21.0,8.36),(-3.0,-6.40)):
        assert through_vital(level,**default)==pytest.approx(measured,abs=0.03)


def test_vital_time_constants():
    assert c.vital_times('band',0.5,0.5)==(1.4,28.0) and c.vital_times('low',0.5,0.5)==(2.8,40.0) and c.vital_times('high',0.5,0.5)==(0.7,15.0)
    attack,release=c.vital_times('band',0.0,1.0)
    assert attack==pytest.approx(5/88.2) and release==pytest.approx(28.0*math.exp(4.0))    # floor: five samples at 88.2 kHz


def test_vital_detector_reading():
    # A sinusoid read 2.2 dB over its mean square at the default times (2.24 at 41 Hz, 2.00 at 65 Hz);
    # the medians over complete patches were 4.5 (mid), 6.3 (high) and 4.3 dB (one band for everything).
    assert [c.detector_db(kind,band,0.5,0.5) for kind,band in (('low','low'),('band','band'),('high','high'),('wide','band'))]==[2.2,4.5,6.3,4.3]
    # Only the ratio of the two times matters: the same reading wherever release - attack is 0.25.
    assert c.detector_db('band','band',0.5,0.75)==pytest.approx(6.1) and c.detector_db('band','band',0.25,0.5)==pytest.approx(6.1)
    assert c.detector_db('high','high',1.0,0.5)==pytest.approx(-2.5)                 # slow attack: under the mean square
    assert c.detector_db('band','band',0.0,0.5)==pytest.approx(7.13,abs=0.02)        # the attack floor counts


# --- mapping ---------------------------------------------------------------------------------------
def same_curve(v,p,reads,levels=range(-70,1,2)):
    for band,delta in reads.items():
        i=c.BANDS.index(band)
        for level in levels:
            vital=c.vital_gain_db(level+delta,*(v[f'{band}_{k}'] for k in ('upper_threshold','upper_ratio','lower_threshold','lower_ratio','gain')))
            assert c.serum_gain_db(level,p,i)==pytest.approx(vital,abs=0.01)


def test_default_settings():
    v=values(); p=params(v)
    assert p['kParamMultiband']==1.0 and p['kParamWet']==100.0 and p['kParamRatio']==1e6 and p['kParamRatioBelow']==1.0
    assert (p['kParamXoverLow'],p['kParamXoverHi'])==(120.0,2500.0) and p['kParamThresh']==0.5 and p['kParamMakeup']==1.0
    assert [p[f'kParamThreshUD{i}'] for i in range(3)]==pytest.approx([84.58,72.50,66.25],abs=0.01)
    assert [p[f'kParamDeadband{i}'] for i in range(3)]==pytest.approx([7.0,11.0,5.0])
    assert [p[f'kParamRatio{i}'] for i in range(3)]==[0.9,0.857,1.0] and [p[f'kParamRatioBelow{i}'] for i in range(3)]==pytest.approx([0.9]*3)
    assert [p[f'kParamGain{i}'] for i in range(3)]==[16.3,11.7,16.3]
    assert p['kParamRelease']==pytest.approx(56.0) and p['kParamAttack']==pytest.approx(2.8)
    same_curve(v,p,dict(low=2.2,band=4.5,high=6.3))
    assert kinds(v)==[('approximation','compressor')]


def test_bands_far_apart_share_the_threshold_scale():
    # The low band is squashed at -78.7 dB, 50 dB under the others.
    v=values(low_upper_threshold=-78.7,low_lower_threshold=-79.0,low_gain=1.0,band_upper_threshold=-28.0)
    p=params(v)
    assert p['kParamThresh']==pytest.approx(0.618,abs=0.001)
    assert [p[f'kParamThreshUD{i}'] for i in range(3)]==pytest.approx([0.0,117.2,128.9],abs=0.1)
    assert p['kParamRatioBelow0']==0.5                       # a lower threshold at the bottom is off
    same_curve(dict(v,low_lower_ratio=0.0),p,dict(low=2.2,band=4.5,high=6.3))
    assert kinds(v)==[('approximation','compressor')]
    # Further apart than ThreshUD reaches: both ends are met half-way and reported.
    wide=values(low_upper_threshold=-80.0,high_upper_threshold=-1.0)
    p=params(wide)
    assert c.serum_knee(p['kParamThresh'],p['kParamThreshUD0'],0)-(-82.2)==pytest.approx(-(c.serum_knee(p['kParamThresh'],p['kParamThreshUD2'],2)-(-7.3)))
    assert ('clipped','compressor_low_upper_threshold') in kinds(wide) and ('clipped','compressor_high_upper_threshold') in kinds(wide)


def test_fewer_bands_move_a_split_out_of_the_way():
    one=params(values(enabled_bands=3))
    assert (one['kParamXoverLow'],one['kParamXoverHi'])==(20.0,20000.0)
    assert (one['kParamRatio0'],one['kParamRatioBelow0'],one['kParamGain0'])==(0.0,0.5,-24.0)      # nothing audible under 20 Hz
    assert (one['kParamRatio2'],one['kParamRatioBelow2'],one['kParamGain2'])==(0.0,0.5,11.7)
    assert c.serum_knee(one['kParamThresh'],one['kParamThreshUD1'],1)==pytest.approx(-25.0-4.3)   # the wide-band reading
    low=params(values(enabled_bands=1)); high=params(values(enabled_bands=2))
    assert (low['kParamXoverLow'],low['kParamXoverHi'])==(120.0,20000.0) and (high['kParamXoverLow'],high['kParamXoverHi'])==(20.0,2500.0)
    assert low['kParamRatio0']==0.9 and low['kParamRatio2']==0.0 and high['kParamRatio2']==1.0 and high['kParamGain0']==-24.0
    assert ('approximation','compressor_enabled_bands') in kinds(values(enabled_bands=1))


def test_negative_lower_ratio_expands_downwards():
    v=values(band_lower_ratio=-0.5); p=params(v)
    assert p['kParamRatioBelow1']==0.25
    same_curve(v,p,dict(band=4.5))


def test_lower_threshold_out_of_reach():
    # 28 dB under the upper one: raised by 4 dB, the upper threshold stays.
    v=values(band_upper_threshold=-8.0); p=params(v)
    assert p['kParamDeadband1']==24.0 and c.serum_knee(p['kParamThresh'],p['kParamThreshUD1'],1)==pytest.approx(-12.5)
    assert ('clipped','compressor_band_lower_threshold') in kinds(v) and p['kParamRatioBelow1']==pytest.approx(0.9)
    # 30 dB under it and close to silence already: left out, which changes less than raising it.
    assert params(values(band_lower_threshold=-55.0))['kParamRatioBelow1']==0.5
    # 66 dB under it: raising it would boost everything quiet, so it is left out.
    v=values(high_upper_threshold=-12.0,high_lower_threshold=-77.8); p=params(v)
    assert p['kParamRatioBelow2']==0.5 and ('clipped','compressor_high_lower_threshold') in kinds(v)
    # No downward compression: the lower threshold is placed exactly instead.
    v=values(band_upper_threshold=-5.0,band_upper_ratio=0.0,band_lower_threshold=-50.0); p=params(v)
    assert c.serum_knee(p['kParamThresh'],p['kParamThreshUD1'],1)-p['kParamDeadband1']==pytest.approx(-54.5)
    assert kinds(v)==[('approximation','compressor')]


def test_gain_beyond_a_band_goes_to_the_makeup():
    v=values(band_gain=30.0); p=params(v)
    assert [p[f'kParamGain{i}'] for i in range(3)]==pytest.approx([10.3,24.0,10.3]) and p['kParamMakeup']==pytest.approx(1.474,abs=0.001)
    same_curve(v,p,dict(low=2.2,band=4.5,high=6.3))
    assert ('clipped','compressor_low_gain') in kinds(values(low_gain=-30.0)) and params(values(low_gain=-30.0))['kParamGain0']==-24.0


def test_times_and_mix():
    assert params(values(mix=0.655))['kParamWet']==pytest.approx(65.5)
    assert params(values(release=0.25))['kParamRelease']==pytest.approx(56.0*math.exp(-2.0))
    slow=values(attack=1.0,release=1.0)
    assert params(slow)['kParamRelease']==1000.0 and ('clipped','compressor_release') in kinds(slow) and ('approximation','compressor_attack') in kinds(slow)


def test_detector_reading_can_be_given():
    p=params(values(),detector=dict(band=2.4))
    assert c.serum_knee(p['kParamThresh'],p['kParamThreshUD1'],1)==pytest.approx(-27.4)
    assert c.serum_knee(p['kParamThresh'],p['kParamThreshUD2'],2)==pytest.approx(-36.3)

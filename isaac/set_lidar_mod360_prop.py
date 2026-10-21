import omni.kit.commands
import omni.usd
from pxr import Gf

stage = omni.usd.get_context().get_stage()
clean_target_path = "/nomad_r_urdf/base_view/sensors_mount/lidar_livox_Mid360"

if stage.GetPrimAtPath(clean_target_path).IsValid():
    omni.kit.commands.execute("DeletePrims", paths=[clean_target_path])

sensor_attributes = {
    'omni:sensor:Core:scanType': "ROTARY",
    'omni:sensor:Core:intensityProcessing': "NORMALIZATION",
    'omni:sensor:Core:rotationDirection': "CW",
    'omni:sensor:Core:rayType': "IDEALIZED",
    'omni:sensor:Core:nearRangeM': 0.1,
    'omni:sensor:Core:farRangeM': 40.0,
    'omni:sensor:Core:rangeResolutionM': 0.004,
    'omni:sensor:Core:rangeAccuracyM': 0.025,
    'omni:sensor:Core:avgPowerW': 0.002,
    'omni:sensor:Core:minReflectance': 0.1,
    'omni:sensor:Core:minReflectanceRange': 70.0,
    'omni:sensor:Core:wavelengthNm': 905.0,
    'omni:sensor:Core:pulseTimeNs': 6,
    'omni:sensor:Core:azimuthErrorMean': 0.1,
    'omni:sensor:Core:azimuthErrorStd': 0.5,
    'omni:sensor:Core:elevationErrorMean': 0.1,
    'omni:sensor:Core:elevationErrorStd': 0.5,
    'omni:sensor:Core:maxReturns': 2,

    # Clean 20Hz timing configurations
    'omni:sensor:Core:scanRateBaseHz': 20.0,
    'omni:sensor:Core:reportRateBaseHz': 20480,

    'omni:sensor:Core:numberOfEmitters': 40,
    'omni:sensor:Core:numberOfChannels': 40,
    'omni:sensor:Core:rangeOffset': 0.03,
    'omni:sensor:Core:intensityMappingType': "LINEAR",
    'omni:sensor:Core:emitterState:s001:azimuthDeg': [0]*40,
    'omni:sensor:Core:emitterState:s001:elevationDeg': [
        -7.0, -5.525, -4.050, -2.575, -1.1004, 0.374, 1.849, 3.324, 4.799, 6.274,
        7.7494, 9.2249, 10.699, 12.174, 13.645, 15.1243, 16.5999, 18.074, 19.5499, 21.024,
        22.493, 23.9749, 25.44, 26.924, 28.39, 29.8743, 31.3499, 32.824, 34.29, 35.774,
        37.2486, 38.724, 40.19, 41.674, 43.14, 44.624, 46.09, 47.574, 49.048, 50.524
    ],

    # FIX: Spacing out by 10ns means max fire time is 390ns, safely below the 2000ns clamp limit!
    'omni:sensor:Core:emitterState:s001:fireTimeNs': [i * 10 for i in range(40)],

    'omni:sensor:Core:emitterState:s001:distanceCorrectionM': [0.0]*40,
    'omni:sensor:Core:emitterState:s001:focalDistM': [0.0]*40,
    'omni:sensor:Core:emitterState:s001:focalSlope': [0.0]*40,
    'omni:sensor:Core:emitterState:s001:horOffsetM': [0.0]*40,
    'omni:sensor:Core:emitterState:s001:reportRateDiv': [1.0]*40,
    'omni:sensor:Core:emitterState:s001:vertOffsetM': [0.0]*40,
    'omni:sensor:Core:emitterState:s001:channelId': list(range(1, 41)),
}

s_, sensor = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    path=clean_target_path,
    parent=None,
    config="",
    variant="",
    translation=(0, 0, 0),
    orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    **sensor_attributes,
)

print("LiDAR profile initialized successfully with safe fire times!")
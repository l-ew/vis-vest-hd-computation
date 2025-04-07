from pathlib import Path
here = Path(__file__).resolve()

BASE_PATH = here.parents[1]
EYE_ORIENTATION = -1
STIM_ORIENTATION = -1
DECODED_DIRECTION = 'Decoded_HD'
DECODED_ROTATION = 'Decoded_HD_shift'
TRUE_ROTATION = 'Actual_HD_shift'
MODEL_ROTATION = 'Decoded_HD_shift_(model)'
EYE_VELOCITY = 'Eye_velocity'
EYE_ROTATION = 'Eye_shift'
NEG_EYE_ROTATION = 'Eye_turn'
MODEL_NEG_EYE_ROTATION = 'Eye_turn_(model)'
PLATFORM_STR = 'head'
SCENE_STR = 'scene'
SYNC_STR = 'sync'
NATURAL_ROT_LONG = 'Head-only rotation'
SCENE_ROT_LONG = 'Scene-only rotation'
SYNC_ROT_LONG = 'Head-and-scene rotation'
DARK_ROT_LONG = 'Head rotation in darkness'
HEAD_VAR = r'$\omega_{\text{h}}$'
SCENE_VAR = r'$\omega_{\text{s}}$'
"""
landmarks/landmark_groups.py
============================
MediaPipe 468-landmark index definitions for all facial regions
needed by Action Unit estimators.

MediaPipe Face Mesh landmark map reference:
  https://google.github.io/mediapipe/solutions/face_mesh.html
"""

from typing import List

# ──────────────────────────────────────────────
# Eyes  (6-point EAR format: p1=inner, p4=outer,
#        p2/p3=upper lid, p5/p6=lower lid)
# ──────────────────────────────────────────────
LEFT_EYE: List[int] = [33, 160, 158, 133, 153, 144]
RIGHT_EYE: List[int] = [362, 385, 387, 263, 373, 380]

# Upper eyelid landmark (for lid raiser AU5)
LEFT_UPPER_LID: List[int] = [159, 158, 157, 173]
RIGHT_UPPER_LID: List[int] = [386, 385, 384, 398]

# ──────────────────────────────────────────────
# Eyebrows
# ──────────────────────────────────────────────
LEFT_BROW: List[int] = [70, 63, 105, 66, 107]
RIGHT_BROW: List[int] = [336, 296, 334, 293, 300]

# Inner brow points (used for AU4 brow knit)
LEFT_BROW_INNER: List[int] = [107, 55, 65]
RIGHT_BROW_INNER: List[int] = [336, 285, 295]

# ──────────────────────────────────────────────
# Nose
# ──────────────────────────────────────────────
NOSE_TIP: List[int] = [1]
NOSE_BRIDGE: List[int] = [6, 197, 195, 5]
LEFT_NOSTRIL: List[int] = [129, 64, 98, 97, 2]
RIGHT_NOSTRIL: List[int] = [358, 294, 327, 326, 2]
NOSE_BASE: List[int] = [2, 326, 327, 94, 97, 98]

# ──────────────────────────────────────────────
# Mouth / Lips
# ──────────────────────────────────────────────
UPPER_LIP_CENTER: int = 13
LOWER_LIP_CENTER: int = 14
MOUTH_LEFT_CORNER: int = 61
MOUTH_RIGHT_CORNER: int = 291

UPPER_LIP_OUTER: List[int] = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_OUTER: List[int] = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
UPPER_LIP_INNER: List[int] = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP_INNER: List[int] = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Lip corners for AU12/AU15
LIP_CORNERS: List[int] = [61, 291]

# ──────────────────────────────────────────────
# Jaw & Chin
# ──────────────────────────────────────────────
CHIN: List[int] = [152, 148, 176, 149, 150]
JAW_LOWER: List[int] = [
    234, 93, 132, 58, 172,
    136, 150, 149, 176, 148,
    152, 377, 400, 378, 379,
    365, 397, 288, 361, 323, 454
]

# ──────────────────────────────────────────────
# Cheeks (used for AU6 cheek raiser estimation)
# ──────────────────────────────────────────────
LEFT_CHEEK: List[int] = [117, 118, 119, 120, 121, 128, 126, 142, 36, 205]
RIGHT_CHEEK: List[int] = [346, 347, 348, 349, 350, 357, 355, 371, 266, 425]

# ──────────────────────────────────────────────
# Forehead / glabella (for AU9 nose wrinkle reference)
# ──────────────────────────────────────────────
GLABELLA: List[int] = [8, 9, 107, 336]

# ──────────────────────────────────────────────
# Key single-point landmarks
# ──────────────────────────────────────────────
NOSE_TIP_IDX: int = 1
CHIN_IDX: int = 152
LEFT_EYE_OUTER_IDX: int = 33
RIGHT_EYE_OUTER_IDX: int = 263
LEFT_MOUTH_IDX: int = 61
RIGHT_MOUTH_IDX: int = 291
LEFT_EAR_IDX: int = 234
RIGHT_EAR_IDX: int = 454
LEFT_FOREHEAD_IDX: int = 103
RIGHT_FOREHEAD_IDX: int = 332

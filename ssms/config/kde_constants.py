KDE_NO_DISPLACE_T = [
    "full_ddm_rv",
    "full_ddm",
    "full_ddm_lan",
    "ddm_st",
    "ddm_normal_st",
    "ddm_truncnormt",
    "ddm_rayleight",
    # TRDM's timer can finish before evidence onset t: RT >= min(t, tT).
    # Shifting every RT by t would drop valid early timer responses.
    "trdm_2",
    "trdm_2_guess",
    "trdm_3",
]

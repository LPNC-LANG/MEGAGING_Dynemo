import numpy as np
from scipy import stats

sex_group_dict = {
    'bm_014': 0, # Female
    'ca_001': 0, 
    'ca_019': 1, # Male
    'cc_007': 0,
    'cm_013': 0,
    'dm_022': 0,
    'el_018': 0,
    'gb_020': 1,
    'gh_017': 1,
    'gp_011': 1,
    'gv_005': 0,
    'lf_012': 0,
    'lr_008': 1,
    'pe_009': 1,
    'pl_016': 1,
    'pr_015': 1,
    'ra_003': 0,
    're_002': 0,
    'sg_010': 1
}
sex = np.array(list(sex_group_dict.values()))-0.5 # Centering

TIV_dict = {
    'bm_014': 1530.42,
    'ca_001': 1417.24, 
    'ca_019': 1440.27,
    'cc_007': 1470.88,
    'cm_013': 1353.45,
    'dm_022': 1409.18,
    'el_018': 1361.40,
    'gb_020': 1482.63,
    'gh_017': 1702.60,
    'gp_011': 1633.97,
    'gv_005': 1337.13,
    'lf_012': 1413.97,
    'lr_008': 1582.01,
    'pe_009': 1589.39,
    'pl_016': 1399.34,
    'pr_015': 1491.61,
    'ra_003': 1582.49,
    're_002': 1302.83,
    'sg_010': 1427.94
}
TIV = stats.zscore(np.array(list(TIV_dict.values())))

VF_semantic_dict = {
    'bm_014': 37,
    'ca_001': 38, 
    'ca_019': 41,
    'cc_007': 31,
    'cm_013': 27,
    'dm_022': 46,
    'el_018': 36,
    'gb_020': 41,
    'gh_017': 35,
    'gp_011': 30,
    'gv_005': 37,
    'lf_012': 54,
    'lr_008': 51,
    'pe_009': 33,
    'pl_016': 19,
    'pr_015': 33,
    'ra_003': 33,
    're_002': 29,
    'sg_010': 37
}
VF_semantic = stats.zscore(np.log(np.array(list(VF_semantic_dict.values()))))

VF_lexical_dict = {
    'bm_014': 17,
    'ca_001': 36, 
    'ca_019': 19,
    'cc_007': 27,
    'cm_013': 19,
    'dm_022': 25,
    'el_018': 31,
    'gb_020': 24,
    'gh_017': 30,
    'gp_011': 24,
    'gv_005': 27,
    'lf_012': 33,
    'lr_008': 41,
    'pe_009': 22,
    'pl_016': 31,
    'pr_015': 26,
    'ra_003': 25,
    're_002': 26,
    'sg_010': 26
}
VF_lexical = stats.zscore(np.log(np.array(list(VF_lexical_dict.values()))))

HAD_A_dict = {
    'bm_014': 6,
    'ca_001': 5, 
    'ca_019': 9,
    'cc_007': 5,
    'cm_013': 5,
    'dm_022': 11,
    'el_018': 6,
    'gb_020': 6,
    'gh_017': 7,
    'gp_011': 5,
    'gv_005': 7,
    'lf_012': 4,
    'lr_008': 2,
    'pe_009': 5,
    'pl_016': 6,
    'pr_015': 1,
    'ra_003': 14,
    're_002': 3,
    'sg_010': 7
}
HAD_A = stats.zscore(np.array(list(HAD_A_dict.values())))

HAD_D_dict = {
    'bm_014': 0,
    'ca_001': 1, 
    'ca_019': 2,
    'cc_007': 0,
    'cm_013': 3,
    'dm_022': 2,
    'el_018': 5,
    'gb_020': 0,
    'gh_017': 7,
    'gp_011': 0,
    'gv_005': 0,
    'lf_012': 0,
    'lr_008': 3,
    'pe_009': 1,
    'pl_016': 4,
    'pr_015': 1,
    'ra_003': 4,
    're_002': 2,
    'sg_010': 0
}
HAD_D = stats.zscore(np.array(list(HAD_D_dict.values())))


TMT_A_dict = {
    'bm_014': 18,
    'ca_001': 16,
    'ca_019': 30,
    'cc_007': 27,
    'cm_013': 35,
    'dm_022': 12,
    'el_018': 14,
    'gb_020': 24,
    'gh_017': 37,
    'gp_011': 20,
    'gv_005': 21,
    'lf_012': 35,
    'lr_008': 19,
    'pe_009': 19,
    'pl_016': 15,
    'pr_015': 36,
    'ra_003': 21,
    're_002': 17,
    'sg_010': 29
}
TMT_A = stats.zscore(np.log(np.array(list(TMT_A_dict.values()))))


TMT_B_dict = {
    'bm_014': 28,
    'ca_001': 39,
    'ca_019': 60,
    'cc_007': 41,
    'cm_013': 73,
    'dm_022': 28,
    'el_018': 23,
    'gb_020': 36,
    'gh_017': 34,
    'gp_011': 44,
    'gv_005': 58,
    'lf_012': 112,
    'lr_008': 30,
    'pe_009': 32,
    'pl_016': 54,
    'pr_015': 62,
    'ra_003': 37,
    're_002': 37,
    'sg_010': 86
}
TMT_B = stats.zscore(np.log(np.array(list(TMT_B_dict.values()))))


MILL_A_dict = {
    'bm_014': 40,
    'ca_001': 37, 
    'ca_019': 44,
    'cc_007': 40,
    'cm_013': 34,
    'dm_022': 41,
    'el_018': 39,
    'gb_020': 41,
    'gh_017': 24,
    'gp_011': 31,
    'gv_005': 43,
    'lf_012': 42,
    'lr_008': 41,
    'pe_009': 40,
    'pl_016': 40,
    'pr_015': 36,
    'ra_003': 41,
    're_002': 40,
    'sg_010': 37
}
MILL_A = stats.zscore(np.array(list(MILL_A_dict.values())))


MILL_B_dict = {
    'bm_014': 40,
    'ca_001': 35, 
    'ca_019': 42,
    'cc_007': 38,
    'cm_013': 40,
    'dm_022': 39,
    'el_018': 34,
    'gb_020': 41,
    'gh_017': 40,
    'gp_011': 30,
    'gv_005': 36,
    'lf_012': 41,
    'lr_008': 39,
    'pe_009': 39,
    'pl_016': 40,
    'pr_015': 39,
    'ra_003': 32,
    're_002': 37,
    'sg_010': 37
}
MILL_B = stats.zscore(np.array(list(MILL_B_dict.values())))
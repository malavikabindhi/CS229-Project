from __future__ import division
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFE
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
mpl.use('Agg')
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import os, itertools, subprocess
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import csv

def load_data():
    biomarkers = [
#                 'FDG', 'AV45',
#                 'ABETA_UPENNBIOMK9_04_19_17', 'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17',
#                 'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp',
    
#                 'CEREBELLUMGREYMATTER_UCBERKELEYAV45_10_17_16',
#                 'WHOLECEREBELLUM_UCBERKELEYAV45_10_17_16',
#                 'ERODED_SUBCORTICALWM_UCBERKELEYAV45_10_17_16',
#                 'FRONTAL_UCBERKELEYAV45_10_17_16',
#                 'CINGULATE_UCBERKELEYAV45_10_17_16',
#                 'PARIETAL_UCBERKELEYAV45_10_17_16',
#                 'TEMPORAL_UCBERKELEYAV45_10_17_16',
#                 'SUMMARYSUVR_WHOLECEREBNORM_UCBERKELEYAV45_10_17_16',
#                 'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF_UCBERKELEYAV45_10_17_16',
#                 'SUMMARYSUVR_COMPOSITE_REFNORM_UCBERKELEYAV45_10_17_16',
#                 'SUMMARYSUVR_COMPOSITE_REFNORM_0.79CUTOFF_UCBERKELEYAV45_10_17_16',
#                 'BRAINSTEM_UCBERKELEYAV45_10_17_16',
#                 'BRAINSTEM_SIZE_UCBERKELEYAV45_10_17_16',
#                 'VENTRICLE_3RD_UCBERKELEYAV45_10_17_16',
#                 'VENTRICLE_3RD_SIZE_UCBERKELEYAV45_10_17_16',
#                 'VENTRICLE_4TH_UCBERKELEYAV45_10_17_16',
#                 'VENTRICLE_4TH_SIZE_UCBERKELEYAV45_10_17_16',
#                 'VENTRICLE_5TH_UCBERKELEYAV45_10_17_16',
#                 'VENTRICLE_5TH_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CC_ANTERIOR_UCBERKELEYAV45_10_17_16',
#                 'CC_ANTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CC_CENTRAL_UCBERKELEYAV45_10_17_16',
#                 'CC_CENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CC_MID_ANTERIOR_UCBERKELEYAV45_10_17_16',
#                 'CC_MID_ANTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CC_MID_POSTERIOR_UCBERKELEYAV45_10_17_16',
#                 'CC_MID_POSTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CC_POSTERIOR_UCBERKELEYAV45_10_17_16',
#                 'CC_POSTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CSF_UCBERKELEYAV45_10_17_16',
#                 'CSF_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_BANKSSTS_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_BANKSSTS_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_CAUDALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_CAUDALANTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_CAUDALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_CAUDALMIDDLEFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_CUNEUS_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_CUNEUS_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_ENTORHINAL_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_ENTORHINAL_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_FRONTALPOLE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_FRONTALPOLE_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_FUSIFORM_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_FUSIFORM_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_INFERIORPARIETAL_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_INFERIORPARIETAL_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_INFERIORTEMPORAL_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_INFERIORTEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_INSULA_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_INSULA_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_ISTHMUSCINGULATE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_ISTHMUSCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_LATERALOCCIPITAL_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_LATERALOCCIPITAL_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_LATERALORBITOFRONTAL_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_LATERALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_LINGUAL_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_LINGUAL_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_MEDIALORBITOFRONTAL_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_MEDIALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_MIDDLETEMPORAL_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_MIDDLETEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_PARACENTRAL_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_PARACENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_PARAHIPPOCAMPAL_UCBERKELEYAV45_10_17_16',
#                 'CTX_LH_PARAHIPPOCAMPAL_SIZE_UCBERKELEYAV45_10_17_16',

    
    ######### UCBERKELEYAV45
    'CEREBELLUMGREYMATTER_UCBERKELEYAV45_10_17_16',
    'WHOLECEREBELLUM_UCBERKELEYAV45_10_17_16',
    'ERODED_SUBCORTICALWM_UCBERKELEYAV45_10_17_16',
    'FRONTAL_UCBERKELEYAV45_10_17_16',
    'CINGULATE_UCBERKELEYAV45_10_17_16',
    'PARIETAL_UCBERKELEYAV45_10_17_16',
    'TEMPORAL_UCBERKELEYAV45_10_17_16',
    'SUMMARYSUVR_WHOLECEREBNORM_UCBERKELEYAV45_10_17_16',
    'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF_UCBERKELEYAV45_10_17_16',
    'SUMMARYSUVR_COMPOSITE_REFNORM_UCBERKELEYAV45_10_17_16',
    'SUMMARYSUVR_COMPOSITE_REFNORM_0.79CUTOFF_UCBERKELEYAV45_10_17_16',
    'BRAINSTEM_UCBERKELEYAV45_10_17_16',
    'BRAINSTEM_SIZE_UCBERKELEYAV45_10_17_16',
    'VENTRICLE_3RD_UCBERKELEYAV45_10_17_16',
    'VENTRICLE_3RD_SIZE_UCBERKELEYAV45_10_17_16',
    'VENTRICLE_4TH_UCBERKELEYAV45_10_17_16',
    'VENTRICLE_4TH_SIZE_UCBERKELEYAV45_10_17_16',
    'VENTRICLE_5TH_UCBERKELEYAV45_10_17_16',
    'VENTRICLE_5TH_SIZE_UCBERKELEYAV45_10_17_16',
    'CC_ANTERIOR_UCBERKELEYAV45_10_17_16',
    'CC_ANTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
    'CC_CENTRAL_UCBERKELEYAV45_10_17_16',
    'CC_CENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CC_MID_ANTERIOR_UCBERKELEYAV45_10_17_16',
    'CC_MID_ANTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
    'CC_MID_POSTERIOR_UCBERKELEYAV45_10_17_16',
    'CC_MID_POSTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
    'CC_POSTERIOR_UCBERKELEYAV45_10_17_16',
    'CC_POSTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
    'CSF_UCBERKELEYAV45_10_17_16',
    'CSF_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_BANKSSTS_UCBERKELEYAV45_10_17_16',
    'CTX_LH_BANKSSTS_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_CAUDALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_CAUDALANTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_CAUDALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_CAUDALMIDDLEFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_CUNEUS_UCBERKELEYAV45_10_17_16',
    'CTX_LH_CUNEUS_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_ENTORHINAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_ENTORHINAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_FRONTALPOLE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_FRONTALPOLE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_FUSIFORM_UCBERKELEYAV45_10_17_16',
    'CTX_LH_FUSIFORM_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_INFERIORPARIETAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_INFERIORPARIETAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_INFERIORTEMPORAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_INFERIORTEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_INSULA_UCBERKELEYAV45_10_17_16',
    'CTX_LH_INSULA_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_ISTHMUSCINGULATE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_ISTHMUSCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_LATERALOCCIPITAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_LATERALOCCIPITAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_LATERALORBITOFRONTAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_LATERALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_LINGUAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_LINGUAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_MEDIALORBITOFRONTAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_MEDIALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_MIDDLETEMPORAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_MIDDLETEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PARACENTRAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PARACENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PARAHIPPOCAMPAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PARAHIPPOCAMPAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PARSOPERCULARIS_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PARSOPERCULARIS_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PARSORBITALIS_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PARSORBITALIS_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PARSTRIANGULARIS_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PARSTRIANGULARIS_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PERICALCARINE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PERICALCARINE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_POSTCENTRAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_POSTCENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_POSTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_POSTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PRECENTRAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PRECENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PRECUNEUS_UCBERKELEYAV45_10_17_16',
    'CTX_LH_PRECUNEUS_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_ROSTRALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_ROSTRALANTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_ROSTRALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_ROSTRALMIDDLEFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_SUPERIORFRONTAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_SUPERIORFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_SUPERIORPARIETAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_SUPERIORPARIETAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_SUPERIORTEMPORAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_SUPERIORTEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_SUPRAMARGINAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_SUPRAMARGINAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_TEMPORALPOLE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_TEMPORALPOLE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_TRANSVERSETEMPORAL_UCBERKELEYAV45_10_17_16',
    'CTX_LH_TRANSVERSETEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_LH_UNKNOWN_UCBERKELEYAV45_10_17_16',
    'CTX_LH_UNKNOWN_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_BANKSSTS_UCBERKELEYAV45_10_17_16',
    'CTX_RH_BANKSSTS_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_CAUDALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_CAUDALANTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_CAUDALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_CAUDALMIDDLEFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_CUNEUS_UCBERKELEYAV45_10_17_16',
    'CTX_RH_CUNEUS_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_ENTORHINAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_ENTORHINAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_FRONTALPOLE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_FRONTALPOLE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_FUSIFORM_UCBERKELEYAV45_10_17_16',
    'CTX_RH_FUSIFORM_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_INFERIORPARIETAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_INFERIORPARIETAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_INFERIORTEMPORAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_INFERIORTEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_INSULA_UCBERKELEYAV45_10_17_16',
    'CTX_RH_INSULA_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_ISTHMUSCINGULATE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_ISTHMUSCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_LATERALOCCIPITAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_LATERALOCCIPITAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_LATERALORBITOFRONTAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_LATERALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_LINGUAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_LINGUAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_MEDIALORBITOFRONTAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_MEDIALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_MIDDLETEMPORAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_MIDDLETEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PARACENTRAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PARACENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PARAHIPPOCAMPAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PARAHIPPOCAMPAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PARSOPERCULARIS_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PARSOPERCULARIS_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PARSORBITALIS_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PARSORBITALIS_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PARSTRIANGULARIS_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PARSTRIANGULARIS_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PERICALCARINE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PERICALCARINE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_POSTCENTRAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_POSTCENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_POSTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_POSTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PRECENTRAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PRECENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PRECUNEUS_UCBERKELEYAV45_10_17_16',
    'CTX_RH_PRECUNEUS_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_ROSTRALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_ROSTRALANTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_ROSTRALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_ROSTRALMIDDLEFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_SUPERIORFRONTAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_SUPERIORFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_SUPERIORPARIETAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_SUPERIORPARIETAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_SUPERIORTEMPORAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_SUPERIORTEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_SUPRAMARGINAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_SUPRAMARGINAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_TEMPORALPOLE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_TEMPORALPOLE_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_TRANSVERSETEMPORAL_UCBERKELEYAV45_10_17_16',
    'CTX_RH_TRANSVERSETEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
    'CTX_RH_UNKNOWN_UCBERKELEYAV45_10_17_16',
    'CTX_RH_UNKNOWN_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_ACCUMBENS_AREA_UCBERKELEYAV45_10_17_16',
    'LEFT_ACCUMBENS_AREA_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_AMYGDALA_UCBERKELEYAV45_10_17_16',
    'LEFT_AMYGDALA_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_CAUDATE_UCBERKELEYAV45_10_17_16',
    'LEFT_CAUDATE_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_CEREBELLUM_CORTEX_UCBERKELEYAV45_10_17_16',
    'LEFT_CEREBELLUM_CORTEX_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_CEREBELLUM_WHITE_MATTER_UCBERKELEYAV45_10_17_16',
    'LEFT_CEREBELLUM_WHITE_MATTER_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_CEREBRAL_WHITE_MATTER_UCBERKELEYAV45_10_17_16',
    'LEFT_CEREBRAL_WHITE_MATTER_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_CHOROID_PLEXUS_UCBERKELEYAV45_10_17_16',
    'LEFT_CHOROID_PLEXUS_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16',
    'LEFT_HIPPOCAMPUS_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_INF_LAT_VENT_UCBERKELEYAV45_10_17_16',
    'LEFT_INF_LAT_VENT_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_LATERAL_VENTRICLE_UCBERKELEYAV45_10_17_16',
    'LEFT_LATERAL_VENTRICLE_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_PALLIDUM_UCBERKELEYAV45_10_17_16',
    'LEFT_PALLIDUM_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_PUTAMEN_UCBERKELEYAV45_10_17_16',
    'LEFT_PUTAMEN_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_THALAMUS_PROPER_UCBERKELEYAV45_10_17_16',
    'LEFT_THALAMUS_PROPER_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_VENTRALDC_UCBERKELEYAV45_10_17_16',
    'LEFT_VENTRALDC_SIZE_UCBERKELEYAV45_10_17_16',
    'LEFT_VESSEL_UCBERKELEYAV45_10_17_16',
    'LEFT_VESSEL_SIZE_UCBERKELEYAV45_10_17_16',
    'NON_WM_HYPOINTENSITIES_UCBERKELEYAV45_10_17_16',
    'NON_WM_HYPOINTENSITIES_SIZE_UCBERKELEYAV45_10_17_16',
    'OPTIC_CHIASM_UCBERKELEYAV45_10_17_16',
    'OPTIC_CHIASM_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_ACCUMBENS_AREA_UCBERKELEYAV45_10_17_16',
    'RIGHT_ACCUMBENS_AREA_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_AMYGDALA_UCBERKELEYAV45_10_17_16',
    'RIGHT_AMYGDALA_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_CAUDATE_UCBERKELEYAV45_10_17_16',
    'RIGHT_CAUDATE_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_CEREBELLUM_CORTEX_UCBERKELEYAV45_10_17_16',
    'RIGHT_CEREBELLUM_CORTEX_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_CEREBELLUM_WHITE_MATTER_UCBERKELEYAV45_10_17_16',
    'RIGHT_CEREBELLUM_WHITE_MATTER_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_CEREBRAL_WHITE_MATTER_UCBERKELEYAV45_10_17_16',
    'RIGHT_CEREBRAL_WHITE_MATTER_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_CHOROID_PLEXUS_UCBERKELEYAV45_10_17_16',
    'RIGHT_CHOROID_PLEXUS_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16',
    'RIGHT_HIPPOCAMPUS_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_INF_LAT_VENT_UCBERKELEYAV45_10_17_16',
    'RIGHT_INF_LAT_VENT_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_LATERAL_VENTRICLE_UCBERKELEYAV45_10_17_16',
    'RIGHT_LATERAL_VENTRICLE_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_PALLIDUM_UCBERKELEYAV45_10_17_16',
    'RIGHT_PALLIDUM_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_PUTAMEN_UCBERKELEYAV45_10_17_16',
    'RIGHT_PUTAMEN_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_THALAMUS_PROPER_UCBERKELEYAV45_10_17_16',
    'RIGHT_THALAMUS_PROPER_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_VENTRALDC_UCBERKELEYAV45_10_17_16',
    'RIGHT_VENTRALDC_SIZE_UCBERKELEYAV45_10_17_16',
    'RIGHT_VESSEL_UCBERKELEYAV45_10_17_16',
    'RIGHT_VESSEL_SIZE_UCBERKELEYAV45_10_17_16',
    'WM_HYPOINTENSITIES_UCBERKELEYAV45_10_17_16',
    'WM_HYPOINTENSITIES_SIZE_UCBERKELEYAV45_10_17_16',
                ]
    
    demographic = ['AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY','APOE4']
    
    cognitive_test =  [ 'ADAS11', 'MMSE', 'RAVLT_immediate'] #'CDRSB',]
    
    features = biomarkers + demographic + cognitive_test
    
    
    tadpole2 = pd.read_csv('TADPOLE_D1_D2.csv',
                           usecols=features + ['DX'],
                           low_memory=False)  
    print(tadpole2.shape)
    tadpole2 = tadpole2.dropna()
    print(tadpole2.shape)
    # tadpole2 = collapse_dx(tadpole2)
    print('tadpole train labels preprocssed.')
    print('NL' , np.sum(tadpole2['DX'] == 'NL'))
    print('MCI' , np.sum(tadpole2['DX'] == 'MCI'))
    print('Dementia' , np.sum(tadpole2['DX'] == 'Dementia'))
    print('NL to MCI' , np.sum(tadpole2['DX'] == 'NL to MCI'))
    print('MCI to NL' , np.sum(tadpole2['DX'] == 'MCI to NL'))
    print('MCI to Dementia' , np.sum(tadpole2['DX'] == 'MCI to Dementia'))
   
    return tadpole2

def collapse_dx(raw_data):
  ret = pd.DataFrame.copy(raw_data)
  ret = ret[ret['DX'] != 'NL to MCI']
  ret = ret[ret['DX'] != 'MCI to NL']

  if 'DX' in ret:
    ret['DX'][ret['DX'] == 'MCI to Dementia'] = 'Dementia'
    ret['DX'][ret['DX'] == 'MCI'] = 'Dementia'
  return ret

def preprocess_data(raw_data):
    
    # Drop missing values
    raw_data_cleaned=raw_data.dropna(how='any')

    #raw_data_cleaned=raw_data_cleaned[(raw_data_cleaned!=' ').all(1)]

    # Convert 'DX' to 2 labels only: MCI is considered Dementia
    raw_data_cleaned=conv_binary_opp(raw_data_cleaned)

    # Set some features as categorical
    xcat_p = raw_data_cleaned[['PTGENDER','PTMARRY','APOE4']]
    raw_data_cleaned.drop(['PTGENDER','PTMARRY','APOE4'], axis=1, inplace=True)
    #PTGENDER: 0:Female; 1: Male -- #PTMARRY: 0:Divorced; 1: Married; 2: Never Married 4:Widowed

    y_p = raw_data_cleaned[['DX']]
    raw_data_cleaned.drop(['DX'], axis=1, inplace=True)
    #DX: 0: Dementia, 1:Normal

    le = preprocessing.LabelEncoder()
    xcat=xcat_p.apply(le.fit_transform)
    x=pd.concat([xcat,raw_data_cleaned],axis=1,join='inner')

    # Set 'DX' (Demented or Not) as categorical
    y=y_p.apply(le.fit_transform)
    comb=pd.concat([x,y],axis=1,join='inner')
    clean_comb=clean_data(comb)

    y = clean_comb[['DX']]
    clean_comb.drop(['DX'], axis=1, inplace=True)
    return clean_comb,y

def clean_data(raw_data):
    xnum= raw_data.apply(pd.to_numeric, errors='coerce')
    xnum = xnum.dropna()
    return xnum

def conv_binary(raw_data_cleaned):
    # Converting 'DX' to 2 labels only: MCI is considered Dementia
    raw_data_cleaned=raw_data_cleaned.replace('Dementia to MCI', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('MCI', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('MCI to Dementia', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('NL to MCI', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('MCI to NL', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('NL to Dementia', 'Dementia')
    return raw_data_cleaned


def conv_binary_opp(raw_data_cleaned):
    # Converting 'DX' to 2 labels only: MCI is considered NL
    raw_data_cleaned=raw_data_cleaned.replace('Dementia to MCI', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('MCI', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('MCI to Dementia', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('NL to MCI', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('MCI to NL', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('NL to Dementia', 'Dementia')
    return raw_data_cleaned


def split_data(x,y, train_split):
    # fraction of the data used in the training set
    m=x.shape[0] # number of data points

    x_train=x.iloc[0:int(m*train_split),:]
    y_train=y.iloc[0:int(m*train_split),:]
    x_test=x.iloc[int(m*train_split)+1:m-1,:]
    y_test=y.iloc[int(m*train_split)+1:m-1,:]
    return x_train, y_train, x_test, y_test

def run_PCA_LDA(X,y,xtest,components):
    y=np.ravel(y)
    target_names = ['Dementia', 'NL'] # 'MCI','NL','MCI to Dementia']

    pca = PCA(n_components=components)
    pca1 =  pca.fit(X)
    X_r = pca1.transform(X)
    Xtest_r = pca1.transform(xtest)

    lda = LinearDiscriminantAnalysis(n_components=10)
    lda1= lda.fit(X, y)
    X_r2 = lda1.transform(X)
    # print('xr2', X_r2.shape)
    Xtest_r2 = lda1.transform(xtest)

    x_pca=pd.DataFrame(X_r)
    x_lda=pd.DataFrame(X_r2)
    xtest_pca=pd.DataFrame(Xtest_r)
    xtest_lda=pd.DataFrame(Xtest_r2)
    y=pd.DataFrame(y)
    return x_pca,x_lda,xtest_pca,xtest_lda

def feature_importances(x, clf):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    header_sorted=[]
    for f in range(len(list(x))):
        header_sorted.append(list(x)[indices[f]])
        print("%d. Feature: %s (%f)" % (f + 1, list(x)[indices[f]], importances[indices[f]]))

    # Plot the feature importances
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(x.shape[1]), header_sorted)
    plt.xlim([-1, x.shape[1]])
    plt.savefig('feature_importance_tadpole.png')
    plt.show()

def add_intercept(X_):
    #####################
    m = X_.shape[0]
    X = np.ones((m, 4))
    X[:, 1:4] = X_
    ###################
    return X


def dist(a, b):
    dist = 0
    ################
    dist = np.sum((a-b) * (a-b))
    ################
    return dist



def findNNOutput(X,y, Xtest, ytest):
    
    clf = MLPClassifier(hidden_layer_sizes=(3,2,), alpha=0.0001,
                    solver='lbfgs', verbose=10, tol=1e-4, learning_rate ='invscaling',
                    learning_rate_init=0.01, max_iter =  10000, activation = 'logistic',
                    random_state = 1)
    
    print(clf.out_activation_)
    

    clf.fit(X, y)
    outputs =  clf.predict(X)
    #print(outputs);
    y_pred = clf.predict(Xtest)
    #print("Training set score: %f" % clf.score(X, y))
    #print("Test set score: %f" % clf.score(Xtest, ytest))
    

    
    """# Confusion Matrix
    cnf_matrix=confusion_matrix(y, outputs)
    #DX: 0: Dementia, 1:MCI to Dementia; 2: MCI; 3: NL
    class_names=list(['Dementia','Normal'])
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')
    
    # Confusion Matrix
    cnf_matrix=confusion_matrix(ytest, y_pred)
    #DX: 0: Dementia, 1:MCI to Dementia; 2: MCI; 3: NL
    class_names=list(['Dementia','Normal'])
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')"""
    return clf.score(X, y), clf.score(Xtest, ytest)
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('dementia_Tadpole.png')
    
def hold_out_CV(x, y):
    splits=50
    rs = ShuffleSplit(n_splits=splits, test_size=.3, random_state=123)
    sum_train =0;
    sum_test = 0;
    for train_index, test_index in rs.split(x):
        pca,lda,x_pca_train,x_lda_train, x_pca_test, x_lda_test=run_PCA_LDA(x.iloc[train_index],y.iloc[train_index],x.iloc[test_index], y.iloc[test_index], components=10)
        [training_score, testing_score] = findNNOutput(x_lda_train, y.iloc[train_index], x_lda_test, y.iloc[test_index])
        sum_train = sum_train + training_score;
        sum_test = sum_test + testing_score;
        
    print("Average Training Score : ", sum_train/splits)
    print("Average Testing Score : ", sum_test/splits)
    
    
def kfold_CV(models, x, y):
    # splits=50
    # rs = ShuffleSplit(n_splits=splits, test_size=.3, random_state=0)
    k=5
    rs = KFold(k, shuffle=True, random_state=123)
    maxAcc = 0

    for name, model in models.items():
      print('\n\nModel: ', name)
      sum_train = 0
      sum_dev_test = 0
      for train_index, dev_test_index in rs.split(x):
        #print(train_index)
        x_pca_train,x_lda_train, x_pca_dev_test, x_lda_dev_test = \
          run_PCA_LDA(x.iloc[train_index],y.iloc[train_index], \
                      x.iloc[dev_test_index], components=10)

        model.fit(x_lda_train, y.iloc[train_index])

        predicted_labels = model.predict(x_lda_dev_test)
        training_score = \
           accuracy_score(y.iloc[train_index], model.predict(x_lda_train))
        dev_testing_score = accuracy_score(y.iloc[dev_test_index], predicted_labels)

        cnf_matrix=confusion_matrix(y.iloc[dev_test_index], predicted_labels)
        class_names=list(['Dementia','NL'])
        # plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
        if(maxAcc < dev_testing_score):
            maxAcc = dev_testing_score
            bestModel = model

        sum_train = sum_train + training_score;
        sum_dev_test = sum_dev_test + dev_testing_score;
        # print('train score', training_score, ' dev test score', dev_testing_score)

      print("Average Training Score : ", sum_train/k)
      print("Average Dev Testing Score : ", sum_dev_test/k)
    return bestModel


def main():
    
    raw_data=load_data()
    x,y=preprocess_data(raw_data)
    
    
    x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size = 0.2, random_state = 123)
    ######### KFOLD CROSS VALIDATION ##########
    models =   {
                    'Logistic Activation w/o regularization': MLPClassifier(hidden_layer_sizes=(3,2,), alpha=0,
                    solver='lbfgs', verbose=10, tol=1e-4, learning_rate ='invscaling',
                    learning_rate_init=0.01, max_iter = 1000, activation = 'logistic',
                    random_state = 1),
                                                                            
                    'relu Activation w/0 regularization': MLPClassifier(hidden_layer_sizes=(3,2,), alpha=0,
                    solver='lbfgs', verbose=10, tol=1e-4, learning_rate ='invscaling',
                    learning_rate_init=0.01, max_iter =  1000, activation = 'relu',
                    random_state = 1),
            
                    'Logistic Activation w regularization': MLPClassifier(hidden_layer_sizes=(3,2,), alpha=0.001,
                    solver='lbfgs', verbose=10, tol=1e-4, learning_rate ='invscaling',
                    learning_rate_init=0.01, max_iter =  1000, activation = 'logistic',
                    random_state = 1), 
                    
                                                                                                           
                    'Logistic Activation w regularization smaller NN ': MLPClassifier(hidden_layer_sizes=(2,2,), alpha=0.001,
                    solver='lbfgs', verbose=10, tol=1e-4, learning_rate ='invscaling',
                    learning_rate_init=0.01, max_iter =  1000, activation = 'logistic',
                    random_state = 1), 
                    'Relu Activation w regularization smaller NW ': MLPClassifier(hidden_layer_sizes=(2,2,), alpha=0.0001,
                    solver='lbfgs', verbose=10, tol=1e-4, learning_rate ='invscaling',
                    learning_rate_init=0.01, max_iter =  1000, activation = 'relu',
                    random_state = 1),
                    
                }
    bestModel = kfold_CV(models, x_train, y_train)
    x_pca_train,x_lda_train, x_pca_test, x_lda_test = \
          run_PCA_LDA(x_train,y_train, \
                      x_test, components=10)
    bestModel.fit(x_lda_train, y_train);
    
    ypred = bestModel.predict(x_lda_test)
    testing_score = accuracy_score(y_test, ypred)
    
    print("Testing Accuracy = " , testing_score)
    
    # Confusion Matrix
    cnf_matrix=confusion_matrix(y_test, ypred)
    #DX: 0: Dementia, 1:MCI to Dementia; 2: MCI; 3: NL
    class_names=list(['Dementia','Normal'])
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')
    

    

if __name__ == '__main__':
    main()
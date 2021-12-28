"""Features (Advia parameters) used in the analysis."""

import numpy as np


features_base = ["WBCP(x10E09 cells/L)", "WBCB(x10E09 cells/L)", "RBC(x10E12 cells/L)", "measHGB(m mol/L)",
                 "MCV(fL)", "CHCM(m mol/L)", "RDW(%)", "HDW(m mol/L)",
                 "PLT(x10E09 cells/L)", "MPV(fL)", "PDW(%)", # platelets
                 "#Retic(x10E09 cells/L)",
                 "mico_ct", "macro_ct", "hypo_ct", "hyper_ct" # Red blood cell hemoglobin volume counts
                 ]

features_percentages = ["%NEUT(%)", "%LYM(%)", "%MONO(%)", "%EOS(%)", "%LUC(%)", "%BASO(%)",
                        "%Retic(%)"] # TODO: verify

features_derived = ["HCT(L/L)", "MCH(fmol)", "MCHC(m mol/L)", "PCT(%)",
                    "abs_neuts(x10E09 cells/L)", "abs_lymphs(x10E09 cells/L)", "abs_monos(x10E09 cells/L)",
                    "abs_eos(x10E09 cells/L)", "abs_lucs(x10E09 cells/L)", "abs_basos(x10E09 cells/L)",
                    "micro_pcnt([No Units])", "macro_pcnt([No Units])",
                    "hypo_pcnt([No Units])", "hyper_pcnt([No Units])", # RBC hemoglobin percentage,
                    "H_mean(fmol)", "H_deviation(fmol)"
                    ]

features = features_base + features_percentages + features_derived


def clean_column_names(x):
    """Remove symbols not allowed by Xgboost."""

    # if x is a list
    items = [('[No Units]', 'No Units')]
    for old, new in items:
        if hasattr(x, 'columns'):
            x.columns = np.array([c.replace(old, new) for c in x.columns])
        else:
            x = [c.replace(old, new) for c in x]
    return x

import numpy as np
from scipy.io import loadmat 
import pandas as pd
import sys
from pathlib import Path
import scipy.linalg as la
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE  # repo root
sys.path.append(str(ROOT))


# def load_planet_data():
#     planets = [
#         # name, mass_1e24kg, diameter_km, density_kg_m3, gravity_m_s2, escape_km_s,
#         # rotation_h, length_of_day_h, dist_mkm, perihelion_mkm, aphelion_mkm,
#         # orbital_period_d, orbital_velocity_km_s, orbital_incl_deg, orbital_ecc,
#         # obliquity_deg, mean_temp_c, surface_pressure_bar, number_of_moons, ring_system, global_magnetic_field
#         ("Mercury", 0.330, 4879, 5429, 3.7, 4.3, 1407.6, 4222.6, 57.9, 46.0, 69.8, 88.0, 47.4, 7.0, 0.206, 0.034, 167, 0.0, 0, False, True),
#         ("Venus", 4.87, 12104, 5243, 8.9, 10.4, -5832.5, 2802.0, 108.2, 107.5, 108.9, 224.7, 35.0, 3.4, 0.007, 177.4, 464, 92.0, 0, False, False),
#         ("Earth", 5.97, 12756, 5514, 9.8, 11.2, 23.9, 24.0, 149.6, 147.1, 152.1, 365.2, 29.8, 0.0, 0.017, 23.4, 15, 1.0, 1, False, True),
#         ("Mars", 0.642, 6792, 3934, 3.7, 5.0, 24.6, 24.7, 228.0, 206.7, 249.3, 687.0, 24.1, 1.8, 0.094, 25.2, -65, 0.01, 2, False, False),
#         ("Jupiter", 1898.0, 142984, 1326, 23.1, 59.5, 9.9, 9.9, 778.5, 740.6, 816.4, 4331.0, 13.1, 1.3, 0.049, 3.1, -110, np.nan, 95, True, True),
#         ("Saturn", 568.0, 120536, 687, 9.0, 35.5, 10.7, 10.7, 1432.0, 1357.6, 1506.5, 10747.0, 9.7, 2.5, 0.052, 26.7, -140, np.nan, 274, True, True),
#         ("Uranus", 86.8, 51118, 1270, 8.7, 21.3, -17.2, 17.2, 2867.0, 2732.7, 3001.4, 30589.0, 6.8, 0.8, 0.047, 97.8, -195, np.nan, 28, True, True),
#         ("Neptune", 102.0, 49528, 1638, 11.0, 23.5, 16.1, 16.1, 4515.0, 4471.1, 4558.9, 59800.0, 5.4, 1.8, 0.010, 28.3, -200, np.nan, 16, True, True),
#         ("Pluto", 0.0130, 2376, 1850, 0.7, 1.3, -153.3, 153.3, 5906.4, 4436.8, 7375.9, 90560.0, 4.7, 17.2, 0.244, 119.5, -225, 0.00001, 5, False, np.nan)
#     ]

#     columns = [
#         "name","mass_1e24kg","diameter_km","density_kg_m3","gravity_m_s2","escape_velocity_km_s",
#         "rotation_period_hours","length_of_day_hours","distance_from_sun_1e6_km","perihelion_1e6_km","aphelion_1e6_km",
#         "orbital_period_days","orbital_velocity_km_s","orbital_inclination_deg","orbital_eccentricity",
#         "obliquity_deg","mean_temperature_c","surface_pressure_bar","number_of_moons","ring_system","global_magnetic_field"
#     ]

#     df_planets = pd.DataFrame(planets, columns=columns)

#     raw_path = Path(ROOT / 'data' / "planets_all.csv")
#     df_planets.to_csv(raw_path, index=False)

#     # NA-free subset
#     na_free_cols = [c for c in df_planets.columns if df_planets[c].notna().all()]
#     df_no_na = df_planets[na_free_cols].copy()

#     # convert booleans to integers before scaling
#     for c in df_no_na.select_dtypes(include=["bool"]).columns:
#         df_no_na[c] = df_no_na[c].astype(int)

#     # Z-score numeric features (mean 0, sd 1)
#     numeric_cols = df_no_na.select_dtypes(include=[np.number]).columns
#     means = df_no_na[numeric_cols].mean()
#     stds = df_no_na[numeric_cols].std(ddof=0)  # population SD; switch to ddof=1 if you prefer sample SD
#     stds = stds.replace(0, 1.0)  # guard against division by zero
#     df_no_na[numeric_cols] = (df_no_na[numeric_cols] - means) / stds

#     # Reorder to ensure 'name' first
#     cols = ["name"] + [c for c in df_no_na.columns if c != "name"]
#     df_z = df_no_na[cols]

#     no_na_path = Path(ROOT / 'data' / "planets_no_missing.csv")
#     df_planets_no_na.to_csv(no_na_path, index=False)

#     return df_planets_no_na


# def load_psychometric_data():
#     names = ["Sentences","Vocabulary","Sent.Completion","First.Letters","4.Letter.Words",
#             "Suffixes","Letter.Series","Pedigrees","Letter.Group"]

#     # 9x9 correlation matrix (from the psych package's Thurstone data)
#     corr = [
#         [1.000, 0.828, 0.776, 0.439, 0.432, 0.447, 0.447, 0.541, 0.380],
#         [0.828, 1.000, 0.779, 0.493, 0.464, 0.489, 0.432, 0.537, 0.358],
#         [0.776, 0.779, 1.000, 0.460, 0.425, 0.443, 0.401, 0.534, 0.359],
#         [0.439, 0.493, 0.460, 1.000, 0.674, 0.590, 0.381, 0.350, 0.424],
#         [0.432, 0.464, 0.425, 0.674, 1.000, 0.541, 0.402, 0.367, 0.446],
#         [0.447, 0.489, 0.443, 0.590, 0.541, 1.000, 0.288, 0.320, 0.325],
#         [0.447, 0.432, 0.401, 0.381, 0.402, 0.288, 1.000, 0.555, 0.598],
#         [0.541, 0.537, 0.534, 0.350, 0.367, 0.320, 0.555, 1.000, 0.452],
#         [0.380, 0.358, 0.359, 0.424, 0.446, 0.325, 0.598, 0.452, 1.000],
#     ]

#     df = pd.DataFrame(corr, columns=names)
#     df.insert(0, "name", names)
#     out_path = Path(ROOT / 'data' / "psychometric.csv")
#     df.to_csv(out_path, index=False)
#     return df


def load_fmri_data():
    names = ['image_bagel',
        'image_brick',
        'image_candle',
        'image_clock',
        'image_glass',
        'image_glove',
        'image_helmet',
        'image_kettle',
        'image_knife',
        'image_ladder',
        'image_pedal',
        'image_saddle',
        'image_spade',
        'image_sponge',
        'image_table',
        'image_wheel',
        'text_bagel',
        'text_brick',
        'text_candle',
        'text_clock',
        'text_glass',
        'text_glove',
        'text_helmet',
        'text_kettle',
        'text_knife',
        'text_ladder',
        'text_pedal',
        'text_saddle',
        'text_spade',
        'text_sponge',
        'text_table',
        'text_wheel'
        ]

    a = np.array([0.38365942, 0.33658986, 0.31859911, 0.29808972, 0.3224023, 0.42602224, 0.37384069, 0.31120577, 0.34875965, 0.31970565, 0.35857025, 0.36532242, 0.35291608, 0.30943487, 0.4088163, 0.35436635, 0.3157959, 0.40211281, 0.39414911, 0.35124768, 0.29041107, 0.49595231, 0.32703596, 0.43430263, 0.30354277, 0.32431712, 0.32335989, 0.31068237, 0.32235356, 0.38775498, 0.37401073, 0.34064614, 0.46852911, 0.29005866, 0.35930841, 0.34972533, 0.33482445, 0.44607791, 0.30067009, 0.36044098, 0.35054297, 0.46780848, 0.40660717, 0.35586898, 0.298206, 0.33660109, 0.4475611, 0.32698626, 0.37007769, 0.36586169, 0.40355882, 0.52560357, 0.30387521, 0.35904086, 0.37814055, 0.37914135, 0.50645511, 0.34805984, 0.37793495, 0.36994403, 0.33578252, 0.32183139, 0.29427345, 0.32296704, 0.38577602, 0.29622093, 0.34148527, 0.3063691, 0.35274345, 0.31752502, 0.33553897, 0.36076734, 0.29295141, 0.37303643, 0.3295859, 0.37702671, 0.30051722, 0.31803179, 0.33935567, 0.30397038, 0.4106019, 0.30567882, 0.37523922, 0.30812558, 0.34693256, 0.38417054, 0.28407867, 0.42171425, 0.37907335, 0.35462167, 0.35871634, 0.32830908, 0.44066941, 0.37882407, 0.35056501, 0.352783, 0.35876029, 0.34108032, 0.3693031, 0.41951924, 0.32281385, 0.40969043, 0.2817943, 0.34364457, 0.3839676, 0.33507307, 0.39492724, 0.29456228, 0.41678108, 0.33392649, 0.37952361, 0.29772352, 0.31596104, 0.30334004, 0.31529434, 0.35301654, 0.39524425, 0.44193822, 0.32615342, 0.39122543, 0.2961974, 0.34743089, 0.30330261, 0.31190851, 0.32571936, 0.41895775, 0.33986222, 0.31562947, 0.34113675, 0.34726745, 0.36534566, 0.28120935, 0.34220995, 0.30775225, 0.31612228, 0.46255136, 0.31082, 0.38209386, 0.28680498, 0.32134924, 0.41164036, 0.27537297, 0.33320177, 0.30753193, 0.39152007, 0.36397378, 0.32662793, 0.38243648, 0.31728578, 0.35151498, 0.24932721, 0.45617331, 0.39326436, 0.28884329, 0.34039174, 0.33339579, 0.3717897, 0.38007074, 0.37731453, 0.39588326, 0.35601921, 0.39088614, 0.32699209, 0.33026386, 0.33613591, 0.2947967, 0.32833944, 0.35856873, 0.3290698, 0.3724945, 0.33371614, 0.37762255, 0.41823704, 0.37746227, 0.39287745, 0.33545191, 0.45862312, 0.45368924, 0.38536121, 0.39355385, 0.40878267, 0.4337443, 0.45540747, 0.31067034, 0.49072365, 0.3965818, 0.44696618, 0.38922851, 0.29964255, 0.38010278, 0.41096019, 0.4649724, 0.4258352, 0.37856928, 0.41447601, 0.44292887, 0.38263938, 0.27643898, 0.35873526, 0.28381349, 0.40886014, 0.29649073, 0.28729771, 0.308368, 0.31876813, 0.36284564, 0.30090555, 0.33305492, 0.35274992, 0.33488425, 0.41681135, 0.30617063, 0.32580317, 0.31228993, 0.34894055, 0.42499663, 0.29331324, 0.41720128, 0.37298401, 0.29181027, 0.37921646, 0.40893044, 0.42524246, 0.36282012, 0.36226838, 0.31531993, 0.40184359, 0.39145155, 0.34207857, 0.3982803, 0.32348785, 0.42010876, 0.3315484, 0.3735424, 0.39639976, 0.37790319, 0.31801319, 0.31270397, 0.35164045, 0.36405153, 0.40597667, 0.35728166, 0.39140814, 0.35211213, 0.2980696, 0.41632799, 0.36780166, 0.29582931, 0.28498624, 0.30733261, 0.3353395, 0.34932668, 0.29807706, 0.38795209, 0.3609488, 0.41229117, 0.33246308, 0.31927944, 0.32739513, 0.33040202, 0.37389012, 0.36090297, 0.36870581, 0.42590143, 0.37845215, 0.31739251, 0.39440427, 0.41350172, 0.37732946, 0.39905022, 0.32466516, 0.36867776, 0.4151196, 0.39035861, 0.41877979, 0.35406982, 0.54079077, 0.3668866, 0.34606039, 0.31999223, 0.36579863, 0.35790416, 0.32348279, 0.34433854, 0.4537239, 0.42875251, 0.457368, 0.40561256, 0.31263115, 0.28871059, 0.28677285, 0.38771861, 0.4115757, 0.36986707, 0.46587296, 0.37491792, 0.42863078, 0.35434501, 0.32831253, 0.35021994, 0.31902437, 0.31619527, 0.35999297, 0.34909999, 0.43116244, 0.32785729, 0.32985608, 0.33461299, 0.50209752, 0.34488972, 0.30025798, 0.40200311, 0.37049691, 0.34059968, 0.31648776, 0.54378493, 0.40867373, 0.54510549, 0.39741953, 0.35848022, 0.44709738, 0.37222573, 0.37263514, 0.46732677, 0.39377312, 0.29872868, 0.41548701, 0.34035773, 0.35706753, 0.3354803, 0.36533742, 0.32028373, 0.38848406, 0.50130004, 0.37206401, 0.46393178, 0.38924986, 0.37345367, 0.49485343, 0.36700646, 0.39204357, 0.4439464, 0.34017554, 0.35041571, 0.29235121, 0.29271388, 0.33851551, 0.31216012, 0.3326029, 0.33529686, 0.39110598, 0.28462362, 0.33179157, 0.3054615, 0.27397701, 0.33504062, 0.31918687, 0.33544426, 0.34311498, 0.29187141, 0.3324812, 0.42111988, 0.35822678, 0.37622419, 0.46223548, 0.40320291, 0.39032707, 0.37070938, 0.34880572, 0.3959683, 0.34416842, 0.42934844, 0.3729773, 0.38004958, 0.40896177, 0.32917717, 0.28309584, 0.34408797, 0.28961083, 0.37933243, 0.35598857, 0.38524887, 0.30824041, 0.35186346, 0.29036155, 0.25686452, 0.35340966, 0.34832248, 0.29922168, 0.40167345, 0.34207291, 0.43519466, 0.36018369, 0.34207539, 0.33114787, 0.44156591, 0.34186326, 0.44250057, 0.31142469, 0.29800567, 0.36722042, 0.35709416, 0.30565547, 0.44562177, 0.35638909, 0.34520724, 0.31360234, 0.33500902, 0.48743093, 0.30587233, 0.42981415, 0.35447747, 0.36849816, 0.48230638, 0.32187483, 0.43022734, 0.32161206, 0.39175851, 0.40721461, 0.34632074, 0.37387487, 0.35499979, 0.29983164, 0.28657787, 0.36146499, 0.42473921, 0.38103872, 0.41971482, 0.35763499, 0.41713811, 0.32697159, 0.55474069, 0.31876417, 0.53949583, 0.34510411, 0.37229907, 0.52457296, 0.33188449, 0.39526109, 0.38775345, 0.34865844, 0.47828601, 0.33477541, 0.39858296, 0.28591541, 0.31457465, 0.34238576, 0.24252126, 0.33733686, 0.34419159, 0.42300143, 0.4494506, 0.40790177, 0.4483051, 0.34847699, 0.40138228, 0.4870107, 0.44446131, 0.45686683, 0.40143464, 0.37959466, 0.30124384, 0.32695512, 0.42228108, 0.30600138, 0.36100341, 0.35246728, 0.33561192, 0.33099148, 0.37227684, 0.38803882, 0.4121324, 0.43871247, 0.37961855, 0.44618495, 0.30766322, 0.37480181, 0.32578568, 0.33469076, 0.34947416, 0.42182684, 0.29100202, 0.31059335, 0.3071345, 0.34933706, 0.36649643, 0.3898815, 0.35702013, 0.45063323, 0.44594839, 0.39050003, 0.30787676, 0.33413993, 0.45483421, 0.40807713, 0.42662313])
    mat = np.zeros((32, 32))
    upper_indices = np.triu_indices(32, k=1)
    mat[upper_indices] = a
    mat.T[upper_indices] = a

    df = pd.DataFrame(mat, columns=names)
    df.insert(0, "name", names)

    df = df.drop(columns=['text_helmet'])
    df = df[df['name'] != 'text_helmet'].reset_index(drop=True)

    out_path = Path(ROOT / 'data' / "fmri.csv")
    df.to_csv(out_path, index=False)
    return df


def load_feature_data(file_name: str) -> pd.DataFrame:
    fp = Path(ROOT / 'data' / file_name)
    if not fp.exists():
        raise FileNotFoundError(f'File not found: {fp}')
    raw = loadmat(fp, struct_as_record=False, squeeze_me=True)
    # filter out MATLAB metadata
    vars_ = {k: v for k, v in raw.items() if not k.startswith('__')}

    # identify data array
    if 'data' in vars_:
        data = vars_['data']
    else:
        # pick the largest 2D numeric array
        candidates = [v for v in vars_.values()
                      if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number)]
        if not candidates:
            raise ValueError("No 2D numeric array found in .mat file to use as data.")
        # choose by maximum size
        data = max(candidates, key=lambda x: x.shape[0] * x.shape[1])

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected data matrix 2D, got shape {data.shape}")
    n_entities, n_feats = data.shape

    # extract or generate names
    if 'names' in vars_:
        raw_names = np.atleast_1d(vars_['names'])
        try:
            names_list = raw_names.tolist()
        except:
            names_list = raw_names
        names_clean = [str(n).strip() for n in names_list]
        if len(names_clean) != n_entities:
            raise ValueError(f"Length of 'names' ({len(names_clean)}) != number of rows in data ({n_entities})")
    else:
        names_clean = [f"entity_{i}" for i in range(n_entities)]

    # extract or generate feature labels
    if 'features' in vars_:
        raw_feats = np.atleast_1d(vars_['features'])
        try:
            feats_list = raw_feats.tolist()
        except:
            feats_list = raw_feats
        features_clean = [str(f).strip() for f in feats_list]
        if len(features_clean) != n_feats:
            raise ValueError(f"Length of 'features' ({len(features_clean)}) != number of columns in data ({n_feats})")
    else:
        features_clean = [f"feature_{j}" for j in range(n_feats)]

    # build DataFrame
    df = pd.DataFrame(data, columns=features_clean)
    df.insert(0, 'name', names_clean)

    # save
    base_name = fp.stem
    df.to_csv(f"data/{base_name}.csv", index=False)

    return df


def load_sim_data(file_name: str) -> pd.DataFrame:
    fp = Path(ROOT / 'data' / file_name)
    if not fp.exists():
        raise FileNotFoundError(f'File not found: {fp}')
    raw = loadmat(fp, struct_as_record=False, squeeze_me=True)
    # filter out MATLAB metadata
    vars_ = {k: v for k, v in raw.items() if not k.startswith('__')}

    # identify the similarity/dissimilarity matrix
    candidates = [
        v for v in vars_.values()
        if isinstance(v, np.ndarray) and v.ndim == 2
        and v.shape[0] == v.shape[1]
        and np.issubdtype(v.dtype, np.number)
    ]
    if not candidates:
        raise ValueError("No square numeric array found in .mat file.")
    S = max(candidates, key=lambda x: x.shape[0])  # largest square matrix

    n_entities = S.shape[0]

    # names
    if "names" in vars_:
        raw_names = np.atleast_1d(vars_["names"])
        try:
            names_list = raw_names.tolist()
        except Exception:
            names_list = raw_names
        names_clean = [str(n).strip() for n in names_list]
        if len(names_clean) != n_entities:
            raise ValueError(f"Length of 'names' ({len(names_clean)}) != matrix size ({n_entities})")
    else:
        names_clean = [f"entity_{i}" for i in range(n_entities)]

    # build DataFrame
    col_labels = [f"entity_{i}" for i in range(n_entities)]
    df = pd.DataFrame(S, columns=col_labels)
    df.insert(0, "name", names_clean)

    # save
    base_name = fp.stem
    df.to_csv(f"data/{base_name}.csv", index=False)

    return df


def dataframe_to_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric = df.select_dtypes(include=[np.number])
    dropped = list(set(df.columns) - set(numeric.columns))
    if dropped:
        warnings.warn(f"Dropping non-numeric columns: {dropped}")
    A = numeric.to_numpy(dtype=float)
    if A.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {A.shape}")
    return A


def centre_and_scale_features(D: np.ndarray) -> np.ndarray:
    """
    Subtract overall mean so the mean of all entries is zero.
    Scale so that the largest eigenvalue of (1/m) D D^T is 1.
    """
    # centring
    mean = np.nanmean(D)
    D_centred = D - mean

    # max variance = 1
    m = D.shape[1] # number of features
    cov = (D_centred @ D_centred.T) / m
    max_eig = np.max(np.linalg.eigvalsh(cov))
    scale = 1.0 / np.sqrt(max_eig)

    return D_centred * scale


def preprocess_similarity_matrix(cov: np.ndarray) -> np.ndarray:
    """Force symmetric matrix to be PSD by zeroing negative eigenvalues."""
    mat = 0.5 * (cov + cov.T)
    w, v = la.eigh(mat)
    w_clipped = np.clip(w, a_min=0.0, a_max=None)
    return (v * w_clipped) @ v.T
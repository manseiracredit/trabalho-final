# Gerar Dataset com dados Sinteticos seguindo os mesmos padroes do Dataset Real 

# .describe() do dataset real para ser usado de base para o dataset falso:

#       obj_ID	alpha	delta	u	g	r	i	z	run_ID	rerun_ID	cam_col	field_ID	spec_obj_ID	redshift	plate	MJD	fiber_ID
# count	1.000000e+05	100000.000000	100000.000000	100000.000000	100000.000000	100000.000000	100000.000000	100000.000000	100000.000000	100000.0	100000.000000	100000.000000	1.000000e+05	100000.000000	100000.000000	100000.000000	100000.000000
# mean	1.237665e+18	177.629117	24.135305	21.980468	20.531387	19.645762	19.084854	18.668810	4481.366060	301.0	3.511610	186.130520	5.783882e+18	0.576661	5137.009660	55588.647500	449.312740
# std	8.438560e+12	96.502241	19.644665	31.769291	31.750292	1.854760	1.757895	31.728152	1964.764593	0.0	1.586912	149.011073	3.324016e+18	0.730707	2952.303351	1808.484233	272.498404
# min	1.237646e+18	0.005528	-18.785328	-9999.000000	-9999.000000	9.822070	9.469903	-9999.000000	109.000000	301.0	1.000000	11.000000	2.995191e+17	-0.009971	266.000000	51608.000000	1.000000
# 25%	1.237659e+18	127.518222	5.146771	20.352353	18.965230	18.135828	17.732285	17.460677	3187.000000	301.0	2.000000	82.000000	2.844138e+18	0.054517	2526.000000	54234.000000	221.000000
# 50%	1.237663e+18	180.900700	23.645922	22.179135	21.099835	20.125290	19.405145	19.004595	4188.000000	301.0	4.000000	146.000000	5.614883e+18	0.424173	4987.000000	55868.500000	433.000000
# 75%	1.237668e+18	233.895005	39.901550	23.687440	22.123767	21.044785	20.396495	19.921120	5326.000000	301.0	5.000000	241.000000	8.332144e+18	0.704154	7400.250000	56777.000000	645.000000
# max	1.237681e+18	359.999810	83.000519	32.781390	31.602240	29.571860	32.141470	29.383740	8162.000000	301.0	6.000000	989.000000	1.412694e+19	7.011245	12547.000000	58932.000000	1000.000000


import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000  # QUANTIDADE DE ENTRADAS NO DATASET FALSO

def normal_clip(mean, std, low, high, size):
    x = np.random.normal(mean, std, size)
    return np.clip(x, low, high)

# IDs
obj_ID = 1_237_000_000_000_000_000 + np.arange(N)
spec_obj_ID = 5_780_000_000_000_000_000 + np.arange(N)

# alpha [0, 360)
alpha = normal_clip(
    mean=177.629117,
    std=96.502241,
    low=0.0,
    high=360.0,
    size=N
)

# delta [-20, 85]
delta = normal_clip(
    mean=24.135305,
    std=19.644665,
    low=-20.0,
    high=85.0,
    size=N
)

# Magnitudes
u = normal_clip(mean=22.2, std=1.5, low=18.0, high=26.0, size=N)
g = normal_clip(mean=20.7, std=1.5, low=17.0, high=25.0, size=N)
r = normal_clip(mean=19.6, std=1.0, low=16.0, high=24.0, size=N)
i = normal_clip(mean=19.1, std=1.0, low=15.5, high=23.5, size=N)
z = normal_clip(mean=18.7, std=1.0, low=15.0, high=23.0, size=N)

# run_ID  [109, 8162]
run_ID = np.random.randint(109, 8163, size=N)

# rerun_ID
rerun_ID = np.full(N, 301, dtype=int)

# cam_col[1, 6]
cam_col = np.random.randint(1, 7, size=N)

# field_ID [11, 989]
field_ID = np.random.randint(11, 990, size=N)

# redshift N(0.57, 0.73) truncado para [0, 7]
redshift = normal_clip(
    mean=0.576661,
    std=0.730707,
    low=0.0,
    high=7.0,
    size=N
)

# plate [266, 12547]
plate = np.random.randint(266, 12548, size=N)

# MJD [51608, 58932]
MJD = np.random.randint(51608, 58933, size=N)

# fiber_ID [1, 1000]
fiber_ID = np.random.randint(1, 1001, size=N)

# monta o DataFrame final
df_fake = pd.DataFrame({
    "obj_ID": obj_ID,
    "alpha": alpha,
    "delta": delta,
    "u": u,
    "g": g,
    "r": r,
    "i": i,
    "z": z,
    "run_ID": run_ID,
    "rerun_ID": rerun_ID,
    "cam_col": cam_col,
    "field_ID": field_ID,
    "spec_obj_ID": spec_obj_ID,
    "redshift": redshift,
    "plate": plate,
    "MJD": MJD,
    "fiber_ID": fiber_ID,
})

# exporta como CSV
df_fake.to_csv("fake_dataset.csv", index=False)

print(df_fake.head())
print(df_fake.describe())

import pandas as pd
from scipy.io import arff
inp = r"C:\Users\waing\OneDrive - Politechnika Warszawska\Dokumenty\ThesisProject\LazyVFDT_v2\airlines.arff\airlines.arff"
out = r"airlines.csv"
data, meta = arff.loadarff(inp)
df = pd.DataFrame(data)
for col in df.columns:
    if df[col].dtype == object:
        try: df[col] = df[col].str.decode("utf-8")
        except AttributeError: pass
df.to_csv(out, index=False)
print("Wrote", out)

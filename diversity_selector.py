import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

# === PARAMETERS ===
INPUT_CSV = "TeamTesseractalGene.csv"     # Replace with your file
OUTPUT_CSV = "TeamTesseractalGene_Diverse.csv"
SCORE_COL = "Score"
SMILES_COL = "SMILES"
ID_COL = "RandomID"
CUTOFF = 0.3  # Tanimoto distance cutoff for Butina

# === LOAD CSV ===
df = pd.read_csv(INPUT_CSV)
print(f"📥 Loaded {df.shape[0]} molecules from {INPUT_CSV}")

# === Generate RDKit Molecules and Fingerprints ===
mols = [Chem.MolFromSmiles(smi) for smi in df[SMILES_COL]]
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]

# === Compute Distance Matrix ===
dists = []
nfps = len(fps)
for i in range(1, nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - x for x in sims])

# === Butina Clustering ===
clusters = Butina.ClusterData(dists, nfps, cutoff=CUTOFF, isDistData=True)
print(f"🧬 Total clusters formed: {len(clusters)}")

# === Assign Cluster IDs ===
df["ClusterID"] = -1
for idx, cluster in enumerate(clusters):
    for mol_idx in cluster:
        df.at[mol_idx, "ClusterID"] = idx

# === Select Top Scoring Compound per Cluster ===
cluster_top = (
    df.sort_values(SCORE_COL, ascending=False)
      .groupby("ClusterID")
      .first()
      .reset_index()
)

# === Select Top N Most Diverse Compounds ===
top_50 = cluster_top.sort_values(SCORE_COL, ascending=False).head(50)
top_200 = cluster_top.sort_values(SCORE_COL, ascending=False).head(200)
top_500 = cluster_top.sort_values(SCORE_COL, ascending=False).head(500)

# === Initialize Columns ===
df["Sel_50"] = 0
df["Sel_200"] = 0
df["Sel_500"] = 0

df.loc[df[ID_COL].isin(top_50[ID_COL]), "Sel_50"] = 1
df.loc[df[ID_COL].isin(top_200[ID_COL]), "Sel_200"] = 1
df.loc[df[ID_COL].isin(top_500[ID_COL]), "Sel_500"] = 1

# === Save Final Submission ===
submission = df[[ID_COL, "Sel_50", "Sel_200", "Sel_500", SCORE_COL]]
submission.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Diversity-aware submission saved to: {OUTPUT_CSV}")


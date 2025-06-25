import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# === INPUT / OUTPUT ===
input_path = "Step2_TestData_Target2035.parquet"
output_path = "Filtered-Step2TestDataTarget2035.csv"

# === LOAD DATA ===
df = pd.read_parquet(input_path)
print(f"Initial size: {df.shape}")

# === REMOVE DUPLICATES ===
df = df.drop_duplicates(subset=["SMILES"])
print(f"After deduplication: {df.shape}")


# === LIPINSKI FILTER ===
def passes_lipinski(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    return (
            mw <= 500 and
            logp <= 5 and
            hbd <= 5 and
            hba <= 10
    )


df["Lipinski"] = df["SMILES"].apply(passes_lipinski)
df = df[df["Lipinski"]].drop(columns=["Lipinski"])
print(f"After Lipinski filtering: {df.shape}")


# === REMOVE INVALID SMILES ===
def is_valid_smiles(smi):
    return Chem.MolFromSmiles(smi) is not None


df = df[df["SMILES"].apply(is_valid_smiles)]
print(f"After removing invalid SMILES: {df.shape}")


# === OPTIONAL: Normalize SMILES (canonicalization) ===
def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


df["SMILES"] = df["SMILES"].apply(canonicalize_smiles)

# === RE-ASSIGN RANDOM ID if needed ===
df = df.reset_index(drop=True)
df["RandomID"] = [f"ID_{i}" for i in range(len(df))]

# === SAVE CLEANED DATA ===
df[["RandomID", "SMILES"]].to_csv(output_path, index=False)
print(f"✅ Saved cleaned test set to: {output_path}")

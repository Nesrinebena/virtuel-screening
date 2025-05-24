import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
data = pd.read_csv("active-compounds.csv")
molecule_smile = data['smilles  ']

form_mol = []
for i in range(len(molecule_smile)):
    smiles = molecule_smile[i]
    mol = Chem.MolFromSmiles(smiles)
    formule_moleculaire = Chem.rdMolDescriptors.CalcMolFormula(mol)
    form_mol.append(formule_moleculaire)

print(form_mol)
data["molecularforma"] = form_mol
data.head()

def compute_morgan_fingerprints_gpu(smiles, radius=2, nBits=124):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=radius, nBits=nBits)
    bit_string = fp.ToBitString()
    return [int(b) for b in bit_string]

data['Morgan_Fingerprints'] = data['smilles  '].apply(compute_morgan_fingerprints_gpu)
print(data['Morgan_Fingerprints'])

data1 = pd.read_csv("compounds-qeury.csv")
molecule_smile1 = data1['smilles']
def compute_morgan_fingerprints_gpu(smiles, radius=2, nBits=124):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=radius, nBits=nBits)
    bit_string = fp.ToBitString()
    return [int(b) for b in bit_string]

data1['Morgan_Fingerprints'] = data1['smilles'].apply(compute_morgan_fingerprints_gpu)
print(data1['Morgan_Fingerprints'])

# Kernel pour calculer la similarité cosinus entre deux vecteurs
cosine_kernel_code = """
__global__ void cosine_similarity_kernel(int *vector1, int *vector2, float *result, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        float dot_product = 0.0;
        float norm1 = 0.0;
        float norm2 = 0.0;

        for (int i = 0; i < %(VECTOR_SIZE)s; ++i)
        {
            dot_product += vector1[idx * %(VECTOR_SIZE)s + i] * vector2[i];
            norm1 += vector1[idx * %(VECTOR_SIZE)s + i] * vector1[idx * %(VECTOR_SIZE)s + i];
            norm2 += vector2[i] * vector2[i];
        }

        result[idx] = dot_product / (sqrtf(norm1) * sqrtf(norm2));
    }
}
"""

# Kernel pour calculer la similarité de Tanimoto entre deux vecteurs
tanimoto_kernel_code = """
__global__ void tanimoto_similarity_kernel(int *vector1, int *vector2, float *result, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        float intersection = 0.0;
        float union_val = 0.0;

        for (int i = 0; i < %(VECTOR_SIZE)s; ++i)
        {
            intersection += fminf(vector1[idx * %(VECTOR_SIZE)s + i], vector2[i]);
            union_val += fmaxf(vector1[idx * %(VECTOR_SIZE)s + i], vector2[i]);
        }

        result[idx] = intersection / union_val;
    }
}
"""

attribue = []
for i in range(len(molecule_smile)):
    s = "cosine_similarity" + str(i)
    attribue.append(s)
attribue1 = []
for i in range(len(molecule_smile)):
    s = "tanimoto_similarity" + str(i)
    attribue1.append(s)
 
# Convert the fingerprints_array to a NumPy array of objects
fingerprints_array = np.array(data['Morgan_Fingerprints'].values, dtype=object)

# Recompile the cosine kernel with the correct data type
cosine_module = SourceModule(cosine_kernel_code % {"VECTOR_SIZE": len(fingerprints_array[0])})
cosine_similarity_kernel = cosine_module.get_function("cosine_similarity_kernel")

# Repeat the same process for the tanimoto kernel
tanimoto_module = SourceModule(tanimoto_kernel_code % {"VECTOR_SIZE": len(fingerprints_array[0])})
tanimoto_similarity_kernel = tanimoto_module.get_function("tanimoto_similarity_kernel")
# Convertir les empreintes en tableaux numpy
fingerprints_array_gpu = cuda.to_device(fingerprints_array)

# Définir la taille des grilles et des blocs
block_size = 256
grid_size = (len(data1) + block_size + 1) // block_size

# Tableaux pour stocker les résultats
cosine_similarity_results = np.zeros(len(data1), dtype=np.float32)
tanimoto_similarity_results = np.zeros(len(data1), dtype=np.float32)
data1_morgan_fingerprints = np.concatenate(data1['Morgan_Fingerprints'].values)
# Appeler les kernelsosine_similarity_kern
cosine_similarity_kernel(cuda.In(data1_morgan_fingerprints.astype(np.int32)), fingerprints_array_gpu, cuda.Out(cosine_similarity_results), np.int32(len(data1)), block=(block_size, 1, 1), grid=(grid_size, 1))
tanimoto_similarity_kernel(cuda.In(data1_morgan_fingerprints.astype(np.int32)),fingerprints_array_gpu, cuda.Out(tanimoto_similarity_results), np.int32(len(data1)), block=(block_size, 1, 1), grid=(grid_size, 1))

# Ajouter les résultats aux données
cosine_similarity_columns = {}
tanimoto_similarity_columns = {}

for i in range(len(attribue)):
    cosine_similarity_columns[attribue[i]] = cosine_similarity_results[i]
    tanimoto_similarity_columns[attribue1[i]] = tanimoto_similarity_results[i]

# Convert dictionaries to DataFrame
cosine_similarity_df = pd.DataFrame(cosine_similarity_columns, index=range(len(cosine_similarity_results)))
tanimoto_similarity_df = pd.DataFrame(tanimoto_similarity_columns, index=range(len(tanimoto_similarity_results)))

# Concatenate DataFrames
data1 = pd.concat([data1, cosine_similarity_df, tanimoto_similarity_df], axis=1)
print(data1.head(10))

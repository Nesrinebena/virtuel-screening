import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import PandasTools
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import time
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def compute_molecular_formula(smiles_list):
    form_mol = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        formule_moleculaire = Chem.rdMolDescriptors.CalcMolFormula(mol)
        form_mol.append(formule_moleculaire)
    return form_mol

def add_molecule_column(df, smiles_col):
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol=smiles_col)
    PandasTools.RenderImagesInAllDataFrames(True)

def compute_morgan_fingerprints(smiles, radius=2, nBits=124):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=radius, nBits=nBits)
    return [int(bit) for bit in fp.ToBitString()]

mod = SourceModule("""
__global__ void compute_similarity(float *matrix, float *vector, float *result, int rows, int cols, int metric) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < rows) {
        float similarity = 0.0;
        float norm_vector = 0.0;
        float norm_row = 0.0;
        float intersection = 0.0;
        float union_val = 0.0;

        if (metric == 0) { // cosine similarity
            for (int j = 0; j < cols; j++) {
                float val = matrix[idx * cols + j];
                float vec_val = vector[j];
                similarity += val * vec_val;
                norm_vector += vec_val * vec_val;
                norm_row += val * val;
            }
            result[idx] = similarity / (sqrt(norm_vector) * sqrt(norm_row));
        } else if (metric == 1) { // Jaccard/Tanimoto similarity
            for (int j = 0; j < cols; j++) {
                float val = matrix[idx * cols + j];
                float vec_val = vector[j];
                intersection += min(val, vec_val);
                union_val += max(val, vec_val);
            }
            result[idx] = intersection / union_val;
        } else if (metric == 2) { // intersection
            float intersection_sum = 0.0;
            for (int j = 0; j < cols; j++) {
                float val = matrix[idx * cols + j];
                float vec_val = vector[j];
                if (val > 0 && vec_val > 0) {
                    intersection_sum += pow(2, j);
                }
            }
            result[idx] = intersection_sum;
        }
    }
}
""")

def calculate_similarity_gpu(matrix, vector, metric):
    matrix = np.array(matrix, dtype=np.float32)
    vector = np.array(vector, dtype=np.float32)
    rows, cols = matrix.shape
    result = np.zeros(rows, dtype=np.float32)

    matrix_gpu = cuda.mem_alloc(matrix.nbytes)
    vector_gpu = cuda.mem_alloc(vector.nbytes)
    result_gpu = cuda.mem_alloc(result.nbytes)

    cuda.memcpy_htod(matrix_gpu, matrix)
    cuda.memcpy_htod(vector_gpu, vector)
    cuda.memcpy_htod(result_gpu, result)

    func = mod.get_function("compute_similarity")
    block_size = 256
    grid_size = (rows + block_size - 1) // block_size
    func(matrix_gpu, vector_gpu, result_gpu, np.int32(rows), np.int32(cols), np.int32(metric), block=(block_size, 1, 1), grid=(grid_size, 1))
    cuda.memcpy_dtoh(result, result_gpu)
    return result

def add_similarity_scores(data, fingerprints_array, metric, prefix):
    new_columns = {f"{prefix}{i}": calculate_similarity_gpu(data['Morgan_Fingerprints'].tolist(), fingerprint, metric)
                   for i, fingerprint in enumerate(fingerprints_array)}
    return pd.concat([data, pd.DataFrame(new_columns)], axis=1)

def calculate_similarity_average(data, attributes):
    total_sum = sum(data[attr] for attr in attributes)
    return total_sum / len(attributes)

def build_and_train_model(x_train, y_train, input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=10, activation='relu', input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Conv1D(filters=64, kernel_size=10, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Conv1D(filters=64, kernel_size=10, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv1D(filters=32, kernel_size=10, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Conv1D(filters=32, kernel_size=10, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv1D(filters=16, kernel_size=10, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Conv1D(filters=16, kernel_size=10, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=448, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(units=400, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(units=300, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(units=200, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(units=180, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(units=1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    model.summary()

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    historique = model.fit(x=x_train, y=y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[early_stopping])
    return model, historique

def main():
    start_time = time.time()

    # Load data
    data = load_data("active-compounds.csv")
    data_test = load_data("lesmoleculerefirencescovid.csv")

    # Compute molecular formulas
    molecule_smile_test = data_test['Smiles']
    data_test["molecularforma"] = compute_molecular_formula(molecule_smile_test)

    # Add molecule columns and compute fingerprints
    add_molecule_column(data_test, 'Smiles')
    data_test['Morgan_Fingerprints'] = data_test['Smiles'].apply(compute_morgan_fingerprints)
    print("fingerprints pour les données de prediction:")
    print(data_test['Morgan_Fingerprints'])
    data_test['Morgan_Fingerprints'].to_csv("fingerprintstest.csv")
    # Process active compounds data
    add_molecule_column(data, 'smilles  ')
    data['Morgan_Fingerprints'] = data['smilles  '].apply(compute_morgan_fingerprints)
    print("----------------------------------------------------------------------------------------------------------------------")
    print("fingerprints pour les molécule refirences:")
    print(data['Morgan_Fingerprints'])
    data['Morgan_Fingerprints'].to_csv("fingerprintactivecomponds.csv")
    # Prepare data for training
    fingerprints_array = np.array(data['Morgan_Fingerprints'].tolist())
    fingerprints_array_test = np.array(data_test['Morgan_Fingerprints'].tolist())
    data1 = load_data("compounds-qeury.csv")
    add_molecule_column(data1, 'smilles')
    data1['Morgan_Fingerprints'] = data1['smilles'].apply(compute_morgan_fingerprints)
    print("----------------------------------------------------------------------------------------------------------------------")
    print("fingerprints pour les molecules candidats:")
    print(data1['Morgan_Fingerprints'])
    data1['Morgan_Fingerprints'].to_csv("fingerprintstest.csv")
    #data1.dropna(inplace=True)
    data1['val_reprintatif'] = data1['Morgan_Fingerprints'].apply(
        lambda x: sum([x[i] * (2 ** i) for i in range(len(x))])
    )
    print("----------------------------------------------------------------------------------------------------------------------")
    print("les valeur reprisentative de chaque molécule candidat")
    print(data1['val_reprintatif'])
    data1['val_reprintatif'].to_csv("valeur reprisentative.csv")
    datanew = data1.copy()

    # Add similarity scores
    data1 = add_similarity_scores(data1, fingerprints_array, metric=0, prefix='cosine_similarity')
    data1 = add_similarity_scores(data1, fingerprints_array, metric=1, prefix='tanimoto_similarity')
    data1 = add_similarity_scores(data1, fingerprints_array, metric=2, prefix='intersaction')
    # Calculate similarity averages
    attributes = [f"cosine_similarity{i}" for i in range(len(data))] + [f"tanimoto_similarity{i}" for i in range(len(data))]
    data1['prob_similarity'] = calculate_similarity_average(data1, attributes)
    data1.head(10)

    # Prepare features and target
    df = data1.drop(columns=['          ', 'smilles', 'molecularforma', 'ROMol', 'Morgan_Fingerprints'])
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X = (X - X.min()) / (X.max() - X.min())
    X = X.astype('float32')
    X.to_csv("X.csv")
    y = y.astype('float32')
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("les données d'entrainement normalizé")
    print(df.head(10))
    print("les information sur notre data d'entrainement ")
    print(df.info())
    # Split data
    x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train model
    model, historique = build_and_train_model(x_train, y_train, input_shape=(X.shape[1], 1))

    # Evaluate model
    loss = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}')
    pd.DataFrame(historique.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    datanew['Morgan_Fingerprints'] = datanew['smilles'].apply(compute_morgan_fingerprints)
    datanew = add_similarity_scores(datanew, fingerprints_array_test, metric=0, prefix='cosine_similarity')
    datanew = add_similarity_scores(datanew, fingerprints_array_test, metric=1, prefix='tanimoto_similarity')
    datanew = add_similarity_scores(datanew, fingerprints_array_test, metric=2, prefix='intersaction')

    datanew.drop(columns=['          '], inplace=True)
    data_final = datanew.copy()
    X = datanew.drop(columns=['smilles', 'molecularforma', 'ROMol', 'Morgan_Fingerprints'])
    X = (X - X.min()) / (X.max() - X.min())
    X = X.astype('float32')
    y_proba = model.predict(X).round(2)
    data_final["score_similarité"] = y_proba
    data_final.sort_values(by="score_similarité", ascending=False, inplace=True)
    data_final['molecularforma'] =compute_molecular_formula(data_final['smilles'])
    print(data_final[['molecularforma', 'score_similarité']].head(40))
    data_final.to_csv("resultatas.csv")

    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"Temps d'exécution : {execution_time:.2f} minute")

if __name__ == "__main__":
    cuda.init()  # Initialize the CUDA context
    main()

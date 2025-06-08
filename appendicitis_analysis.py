import pandas as pd
import requests
import io
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Etapa 1: Download e Extração dos Dados
url = "https://zenodo.org/records/7669442/files/app_data.xlsx?download=1"
data_df = pd.DataFrame()

try:
    print("Baixando dados...")
    r = requests.get(url)
    r.raise_for_status()
    data_df = pd.read_excel(io.BytesIO(r.content))
    print("Dados carregados com sucesso do arquivo Excel.")
except requests.exceptions.RequestException as e:
    print(f"Ocorreu um erro durante o download: {e}")
except Exception as e:
    print(f"Ocorreu um erro durante a extração ou leitura dos dados: {e}")

if not data_df.empty:
    print("\nIniciando pré-processamento dos dados...")
    df_processed = data_df.copy()

    cols_to_drop = ['ID', 'Patient_ID', 'Date', 'US_Image_Count']
    cols_to_drop_existing = [col for col in cols_to_drop if col in df_processed.columns]
    df_processed = df_processed.drop(columns=cols_to_drop_existing)
    print(f"Colunas removidas: {cols_to_drop_existing}")

    print("Tratando valores ausentes...")
    for col in df_processed.select_dtypes(include=np.number).columns:
        if df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            df_processed.loc[:, col] = df_processed[col].fillna(median_val)
            # print(f"Valores ausentes em '{col}' preenchidos com mediana: {median_val}") # Comentado para reduzir output

    for col in df_processed.select_dtypes(include=['object']).columns:
        if df_processed[col].isnull().any():
            mode_val = df_processed[col].mode()[0]
            df_processed.loc[:, col] = df_processed[col].fillna(mode_val)
            # print(f"Valores ausentes em '{col}' preenchidos com moda: {mode_val}") # Comentado para reduzir output

    print("Codificando variáveis categóricas...")
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    targets = ['Management', 'Severity', 'Diagnosis']

    features_categorical_to_encode = [col for col in categorical_cols if col not in targets and col in df_processed.columns]

    if features_categorical_to_encode:
        df_encoded = pd.get_dummies(df_processed, columns=features_categorical_to_encode, drop_first=True)
        # print(f"Variáveis codificadas com one-hot (drop_first=True): {features_categorical_to_encode}") # Comentado
    else:
        df_encoded = df_processed.copy()
        print("Nenhuma feature categórica (não alvo) para codificar com one-hot.")

    # --- Debug: Contagem de classes em Management ANTES do LabelEncoder ---
    if 'Management' in df_processed.columns:
        print("\nContagem de classes em 'Management' (antes do LabelEncoder):")
        print(df_processed['Management'].value_counts())
    # --- Fim do Debug ---

    le_management = LabelEncoder()
    if 'Management' in df_encoded.columns:
        df_encoded.loc[:, 'Management'] = le_management.fit_transform(df_encoded['Management'])
        print("Variável 'Management' codificada com LabelEncoder.")
        # --- Debug: Classes reais em le_management ---
        print(f"Classes reais em le_management: {list(le_management.classes_)}")
        # --- Fim do Debug ---

    le_severity = LabelEncoder()
    if 'Severity' in df_encoded.columns:
        df_encoded.loc[:, 'Severity'] = le_severity.fit_transform(df_encoded['Severity'])
        print("Variável 'Severity' codificada com LabelEncoder.")

    le_diagnosis = LabelEncoder()
    if 'Diagnosis' in df_encoded.columns:
        df_encoded.loc[:, 'Diagnosis'] = le_diagnosis.fit_transform(df_encoded['Diagnosis'])
        print("Variável 'Diagnosis' codificada com LabelEncoder.")

    print("\n--- Treinando Modelo 1: Previsão de Manejo ---")
    model1 = None
    X1_global = pd.DataFrame()

    if 'Management' in df_encoded.columns:
        feature_columns = [col for col in df_encoded.columns if col not in targets]
        X1 = df_encoded[feature_columns].copy()
        y1 = df_encoded['Management'].copy()
        X1_global = X1.copy()

        # print(f"Debug: y1 dtype ANTES da conversão: {y1.dtype}") # Comentado
        # print(f"Debug: y1 NaNs ANTES da conversão: {y1.isnull().sum()}") # Comentado
        # print(f"Debug: y1 unique values ANTES: {y1.unique()[:10]}") # Comentado

        if y1.isnull().any():
            print(f"ALERTA: y1 contém {y1.isnull().sum()} NaNs APÓS LabelEncoding. Isso é inesperado.")

        if pd.api.types.is_float_dtype(y1.dtype):
            # print("Debug: y1 é float. Tentando tratar NaNs e converter para int.") # Comentado
            if y1.isnull().any():
                 if not y1.dropna().empty:
                     y1_mode = y1.mode()[0]
                     y1 = y1.fillna(y1_mode)
                     # print(f"Debug: y1 NaNs (float) preenchidos com moda ({y1_mode}). NaNs restantes: {y1.isnull().sum()}") # Comentado
                 else:
                     raise ValueError("y1 é float, contém apenas NaNs, não é possível calcular a moda.")
            if not y1.isnull().any():
                y1 = y1.astype(int)
            else:
                raise ValueError("y1 é float e ainda contém NaNs após tentativa de preenchimento.")
        elif not pd.api.types.is_integer_dtype(y1.dtype):
            # print(f"Debug: y1 não é int nem float (dtype: {y1.dtype}). Tentando converter para numérico e depois int.") # Comentado
            y1 = pd.to_numeric(y1, errors='coerce')
            if y1.isnull().any():
                # print(f"ALERTA: y1 tem {y1.isnull().sum()} NaNs após pd.to_numeric. Preenchendo com moda.") # Comentado
                if not y1.dropna().empty:
                    y1_mode = y1.mode()[0]
                    y1 = y1.fillna(y1_mode)
                    # print(f"Debug: y1 NaNs (object/other) preenchidos com moda ({y1_mode}). NaNs restantes: {y1.isnull().sum()}") # Comentado
                else:
                    raise ValueError("Todos os valores em y1 se tornaram NaN após to_numeric, não é possível prosseguir.")
            if y1.isnull().any():
                raise ValueError("y1 ainda contém NaNs após tentativa de preenchimento, antes de astype(int).")
            y1 = y1.astype(int)

        # print(f"Debug: y1 dtype APÓS conversão: {y1.dtype}") # Comentado
        # print(f"Debug: y1 NaNs APÓS conversão: {y1.isnull().sum()}") # Comentado
        # print(f"Debug: y1 unique values APÓS: {y1.unique()[:10]}") # Comentado

        model1 = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores1 = cross_val_score(model1, X1, y1, cv=5, scoring='accuracy', error_score='raise')
        print(f"Acurácia (Validação Cruzada) do Modelo de Manejo: {cv_scores1.mean():.4f} (+/- {cv_scores1.std() * 2:.4f})")
        model1.fit(X1, y1)
        print("Modelo de Manejo treinado.")
    else:
        print("Coluna 'Management' não encontrada. Modelo de Manejo não será treinado.")

    print("\n--- Treinando Modelo 2: Previsão de Severidade (Casos Cirúrgicos) ---")
    model2 = None
    surgical_management_codes_global = np.array([])

    if model1 and 'Management' in df_encoded.columns and 'Severity' in df_encoded.columns:
        try:
            # Ajuste os nomes das classes aqui com base na saída do print(list(le_management.classes_))
            # Exemplo: Se a saída mostrar 'Primary Surgical' (com S maiúsculo), ajuste aqui:
            surgical_classes_to_transform = [cls for cls in ['Primary surgical', 'Secondary surgical'] if cls in le_management.classes_]
            
            if not surgical_classes_to_transform:
                 print(f"ALERTA: Nenhuma das classes cirúrgicas esperadas ('Primary surgical', 'Secondary surgical') foi encontrada em le_management.classes_: {list(le_management.classes_)}")

            if surgical_classes_to_transform:
                surgical_management_codes = le_management.transform(surgical_classes_to_transform)
                surgical_management_codes_global = surgical_management_codes
                df_surgical = df_encoded[df_encoded['Management'].isin(surgical_management_codes)].copy()

                if not df_surgical.empty:
                    feature_columns_m2 = [col for col in df_surgical.columns if col not in targets]
                    X2 = df_surgical[feature_columns_m2].copy()
                    y2 = df_surgical['Severity'].copy()

                    # print(f"Debug: y2 dtype ANTES da conversão: {y2.dtype}") # Comentado
                    # print(f"Debug: y2 NaNs ANTES da conversão: {y2.isnull().sum()}") # Comentado
                    if y2.isnull().any():
                        print(f"ALERTA: y2 contém {y2.isnull().sum()} NaNs. Isso é inesperado.")
                    if pd.api.types.is_float_dtype(y2.dtype):
                        # print("Debug: y2 é float. Tentando tratar NaNs e converter para int.") # Comentado
                        if y2.isnull().any():
                            if not y2.dropna().empty:
                                y2_mode = y2.mode()[0]
                                y2 = y2.fillna(y2_mode)
                                # print(f"Debug: y2 NaNs (float) preenchidos com moda ({y2_mode}). NaNs restantes: {y2.isnull().sum()}") # Comentado
                            else: 
                                raise ValueError("y2 é float, contém apenas NaNs.")
                        if not y2.isnull().any(): 
                            y2 = y2.astype(int)
                        else: 
                            raise ValueError("y2 é float e ainda contém NaNs.")
                    elif not pd.api.types.is_integer_dtype(y2.dtype):
                        # print(f"Debug: y2 não é int nem float (dtype: {y2.dtype}). Tentando converter para numérico e depois int.") # Comentado
                        y2 = pd.to_numeric(y2, errors='coerce')
                        if y2.isnull().any():
                            if not y2.dropna().empty:
                                y2_mode = y2.mode()[0]
                                y2 = y2.fillna(y2_mode)
                                # print(f"Debug: y2 NaNs (object/other) preenchidos com moda ({y2_mode}). NaNs restantes: {y2.isnull().sum()}") # Comentado
                            else:
                                raise ValueError("Todos os valores em y2 se tornaram NaN.")
                        if y2.isnull().any(): 
                            raise ValueError("y2 ainda contém NaNs antes de astype(int).")
                        y2 = y2.astype(int)
                    # print(f"Debug: y2 dtype APÓS conversão: {y2.dtype}") # Comentado
                    # print(f"Debug: y2 NaNs APÓS conversão: {y2.isnull().sum()}") # Comentado

                    if len(y2.unique()) > 1:
                        model2 = RandomForestClassifier(n_estimators=100, random_state=42)
                        cv_scores2 = cross_val_score(model2, X2, y2, cv=5, scoring='accuracy', error_score='raise')
                        print(f"Acurácia (Validação Cruzada) do Modelo de Severidade: {cv_scores2.mean():.4f} (+/- {cv_scores2.std() * 2:.4f})")
                        model2.fit(X2, y2)
                        print("Modelo de Severidade treinado.")
                    else:
                        print("Não há variedade de classes suficiente nos casos cirúrgicos para treinar o modelo de severidade.")
                else:
                    print("Não foram encontrados casos cirúrgicos para treinar o modelo de severidade (df_surgical vazio após filtrar por Management).")
            else:
                print("Classes de manejo cirúrgico ('Primary surgical', 'Secondary surgical') não encontradas no LabelEncoder de Management. Modelo 2 não será treinado.")
        except Exception as e:
            print(f"Erro ao treinar modelo de severidade: {e}")
    else:
        print("Modelo de Manejo não treinado ou colunas 'Management'/'Severity' ausentes. Modelo de Severidade não será treinado.")

    def intelligent_appendicitis_system(patient_data_dict):
        print("\n--- Análise do Paciente ---")
        if not model1 or X1_global.empty:
            print("Sistema de decisão não pode operar: Modelo de Manejo não está treinado ou features de referência (X1_global) não estão disponíveis.")
            return

        input_df = pd.DataFrame([patient_data_dict])
        patient_categorical_to_encode = [col for col in features_categorical_to_encode if col in input_df.columns]
        if patient_categorical_to_encode:
            input_encoded = pd.get_dummies(input_df, columns=patient_categorical_to_encode, drop_first=True)
        else:
            input_encoded = input_df.copy()

        input_aligned, _ = input_encoded.align(X1_global, join='right', axis=1, fill_value=0)
        
        for col in input_aligned.columns:
            input_aligned[col] = pd.to_numeric(input_aligned[col], errors='coerce').fillna(0)

        management_pred_code = model1.predict(input_aligned)[0]
        management_pred_label = le_management.inverse_transform([management_pred_code])[0]
        print(f"Previsão de Manejo: {management_pred_label}")

        if model2 and management_pred_code in surgical_management_codes_global:
            severity_pred_code = model2.predict(input_aligned)[0]
            severity_pred_label = le_severity.inverse_transform([severity_pred_code])[0]
            print(f"Previsão de Severidade para Caso Cirúrgico: {severity_pred_label}")
        elif not model2 and management_pred_code in surgical_management_codes_global:
             print("Modelo de severidade não está disponível, mas o manejo é cirúrgico.")
        elif management_pred_code not in surgical_management_codes_global:
            print("A previsão de severidade não se aplica a manejos não cirúrgicos.")
        else:
            print("Modelo de severidade não disponível e manejo não cirúrgico.")

    if not data_df.empty and not X1_global.empty:
        sample_patient_data_raw = data_df.drop(columns=targets, errors='ignore').iloc[0].to_dict()
        print("\n--- Demonstração com Paciente Exemplo (dados brutos antes do pré-processamento interno da função): ---")
        # print(sample_patient_data_raw) # Comentado para reduzir output
        intelligent_appendicitis_system(sample_patient_data_raw)
    elif X1_global.empty:
        print("\nNão é possível rodar a demonstração: X1_global (features de referência) não foi definido, provavelmente devido a erro no treino do Modelo 1.")
    else:
        print("\nNão é possível rodar a demonstração: DataFrame original está vazio.")
else:
    print("Pipeline de análise não executado pois os dados não foram carregados.")

print("\n--- Fim da Análise de Apendicite ---")
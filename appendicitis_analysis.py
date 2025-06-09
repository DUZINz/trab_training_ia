import pandas as pd
import numpy as np
import requests
import io
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Baixa e carrega os dados do Excel
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

    # Remove colunas irrelevantes
    cols_to_drop = ['ID', 'Patient_ID', 'Date', 'US_Image_Count']
    cols_to_drop_existing = [col for col in cols_to_drop if col in df_processed.columns]
    df_processed = df_processed.drop(columns=cols_to_drop_existing)
    print(f"Colunas removidas: {cols_to_drop_existing}")

    # Preenche valores ausentes: mediana para numéricos, moda para categóricos
    print("Tratando valores ausentes...")
    for col in df_processed.select_dtypes(include=np.number).columns:
        if df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)

    for col in df_processed.select_dtypes(include=['object']).columns:
        if df_processed[col].isnull().any():
            mode_val = df_processed[col].mode()[0]
            df_processed[col] = df_processed[col].fillna(mode_val)

    print("Codificando variáveis categóricas...")
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    targets = ['Management', 'Severity', 'Diagnosis']

    # Codifica features categóricas (não-alvo) usando one-hot encoding
    features_categorical_to_encode = [col for col in categorical_cols if col not in targets and col in df_processed.columns]
    if features_categorical_to_encode:
        df_encoded = pd.get_dummies(df_processed, columns=features_categorical_to_encode, drop_first=True)
    else:
        df_encoded = df_processed.copy()

    if 'Management' in df_processed.columns:
        print("\nContagem de classes em 'Management' (antes do LabelEncoder):")
        print(df_processed['Management'].value_counts())

    # Codifica variáveis alvo com LabelEncoder
    le_management = LabelEncoder()
    if 'Management' in df_encoded.columns:
        df_encoded['Management'] = le_management.fit_transform(df_encoded['Management'])
        print("Variável 'Management' codificada com LabelEncoder.")
        print(f"Classes reais em le_management: {list(le_management.classes_)}")

    le_severity = LabelEncoder()
    if 'Severity' in df_encoded.columns:
        df_encoded['Severity'] = le_severity.fit_transform(df_encoded['Severity'])
        print("Variável 'Severity' codificada com LabelEncoder.")

    le_diagnosis = LabelEncoder()
    if 'Diagnosis' in df_encoded.columns:
        df_encoded['Diagnosis'] = le_diagnosis.fit_transform(df_encoded['Diagnosis'])

    # Treina o modelo de manejo (Management)
    print("\n--- Treinando Modelo 1: Previsão de Manejo ---")
    model1 = None
    X1_global = pd.DataFrame()
    if 'Management' in df_encoded.columns:
        feature_columns = [col for col in df_encoded.columns if col not in targets]
        X1 = df_encoded[feature_columns].copy()
        y1 = df_encoded['Management'].copy()
        X1_global = X1.copy()
        # Garante que y1 seja inteiro e sem NaNs
        y1 = y1.fillna(y1.mode()[0]).astype(int)
        model1 = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores1 = cross_val_score(model1, X1, y1, cv=5, scoring='accuracy', error_score='raise')
        print(f"Acurácia (Validação Cruzada) do Modelo de Manejo: {cv_scores1.mean():.4f} (+/- {cv_scores1.std() * 2:.4f})")
        model1.fit(X1, y1)
        print("Modelo de Manejo treinado.")

    # Treina o modelo de severidade (Severity) apenas para casos cirúrgicos
    print("\n--- Treinando Modelo 2: Previsão de Severidade (Casos Cirúrgicos) ---")
    model2 = None
    surgical_management_codes_global = np.array([])
    if model1 and 'Management' in df_encoded.columns and 'Severity' in df_encoded.columns:
        # Identifica códigos de manejo cirúrgico
        surgical_classes_to_transform = [cls for cls in ['primary surgical', 'secondary surgical'] if cls in le_management.classes_]
        if surgical_classes_to_transform:
            surgical_management_codes = le_management.transform(surgical_classes_to_transform)
            surgical_management_codes_global = surgical_management_codes
            df_surgical = df_encoded[df_encoded['Management'].isin(surgical_management_codes)].copy()
            if not df_surgical.empty:
                feature_columns_m2 = [col for col in df_surgical.columns if col not in targets]
                X2 = df_surgical[feature_columns_m2].copy()
                y2 = df_surgical['Severity'].copy()
                y2 = y2.fillna(y2.mode()[0]).astype(int)
                if len(y2.unique()) > 1:
                    model2 = RandomForestClassifier(n_estimators=100, random_state=42)
                    cv_scores2 = cross_val_score(model2, X2, y2, cv=5, scoring='accuracy', error_score='raise')
                    print(f"Acurácia (Validação Cruzada) do Modelo de Severidade: {cv_scores2.mean():.4f} (+/- {cv_scores2.std() * 2:.4f})")
                    model2.fit(X2, y2)
                    print("Modelo de Severidade treinado.")

    # Sistema de apoio à decisão: recebe dados de um paciente e faz as previsões
    def intelligent_appendicitis_system(patient_data_dict):
        # Pré-processa os dados do paciente para alinhar com as features do modelo
        input_df = pd.DataFrame([patient_data_dict])
        patient_categorical_to_encode = [col for col in features_categorical_to_encode if col in input_df.columns]
        if patient_categorical_to_encode:
            input_encoded = pd.get_dummies(input_df, columns=patient_categorical_to_encode, drop_first=True)
        else:
            input_encoded = input_df.copy()
        input_aligned, _ = input_encoded.align(X1_global, join='right', axis=1, fill_value=0)
        for col in input_aligned.columns:
            input_aligned[col] = pd.to_numeric(input_aligned[col], errors='coerce').fillna(0)

        # Previsão de Manejo
        management_pred_code = model1.predict(input_aligned)[0]
        management_pred_label = le_management.inverse_transform([management_pred_code])[0]
        print(f"Previsão de Manejo: {management_pred_label}")

        # Previsão de Severidade se manejo for cirúrgico
        if model2 and management_pred_code in surgical_management_codes_global:
            severity_pred_code = model2.predict(input_aligned)[0]
            severity_pred_label = le_severity.inverse_transform([severity_pred_code])[0]
            print(f"Previsão de Severidade para Caso Cirúrgico: {severity_pred_label}")
        elif not model2 and management_pred_code in surgical_management_codes_global:
            print("Modelo de severidade não está disponível, mas o manejo é cirúrgico.")
        elif management_pred_code not in surgical_management_codes_global:
            print("A previsão de severidade não se aplica a manejos não cirúrgicos.")

    # Demonstração do sistema com um paciente exemplo
    if not data_df.empty and not X1_global.empty:
        sample_patient_data_raw = data_df.drop(columns=targets, errors='ignore').iloc[0].to_dict()
        print("\n--- Demonstração com Paciente Exemplo (dados brutos antes do pré-processamento interno da função): ---")
        intelligent_appendicitis_system(sample_patient_data_raw)
    elif X1_global.empty:
        print("\nNão é possível rodar a demonstração: X1_global (features de referência) não foi definido, provavelmente devido a erro no treino do Modelo 1.")
    else:
        print("\nNão é possível rodar a demonstração: DataFrame original está vazio.")
else:
    print("Pipeline de análise não executado pois os dados não foram carregados.")

print("\n--- Fim da Análise de Apendicite ---")
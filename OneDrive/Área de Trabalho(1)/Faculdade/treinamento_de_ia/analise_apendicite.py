import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os # Importe a biblioteca 'os' para verificar a existência do arquivo

def main():
    """Função principal para organizar o código."""
    
    # --- Carregamento dos Dados ---
    # Prioriza o arquivo local, se não encontrar, baixa da internet.
    arquivo_local = 'app_data.csv'
    if os.path.exists(arquivo_local):
        print(f"Carregando dados do arquivo local: {arquivo_local}")
        df = pd.read_csv(arquivo_local)
    else:
        print("Arquivo local não encontrado. Baixando dados da internet...")
        try:
            url = "https://raw.githubusercontent.com/i6092467/pediatric-appendicitis-ml/main/app_data.csv"
            df = pd.read_csv(url)
            # Opcional: Salvar uma cópia local para uso futuro
            df.to_csv(arquivo_local, index=False)
            print(f"Dados baixados e salvos como '{arquivo_local}' para uso futuro.")
        except Exception as e:
            print(f"Não foi possível carregar os dados. Erro: {e}")
            return # Encerra o script se não conseguir os dados

    # --- 1. Pré-processamento dos Dados ---
    df_processed = df.drop(columns=['ID', 'Patient_ID', 'Date', 'US_Image_Count', 'US_Number'])
    targets = ['Management', 'Severity', 'Diagnosis']
    features = df_processed.drop(columns=targets)
    
    numeric_features = features.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = features.select_dtypes(include=['object', 'bool']).columns

    imputer_numeric = SimpleImputer(strategy='median')
    features[numeric_features] = imputer_numeric.fit_transform(features[numeric_features])
    
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    features[categorical_features] = imputer_categorical.fit_transform(features[categorical_features])
    
    features_encoded = pd.get_dummies(features, columns=categorical_features, drop_first=True)

    le_management = LabelEncoder()
    le_severity = LabelEncoder()
    
    y1_management = le_management.fit_transform(df_processed['Management'])
    y2_severity = le_severity.fit_transform(df_processed['Severity'])

    # --- 2. Modelo 1: Previsão de Manejo ---
    print("\n--- Treinando Modelo 1: Previsão de Manejo ---")
    model_management = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    cv_scores_management = cross_val_score(model_management, features_encoded, y1_management, cv=5, scoring='accuracy')
    print(f"Acurácia Média (Manejo): {cv_scores_management.mean():.4f}")
    model_management.fit(features_encoded, y1_management)

    # --- 3. Modelo 2: Previsão de Gravidade (Casos Cirúrgicos) ---
    print("\n--- Treinando Modelo 2: Previsão de Gravidade (Casos Cirúrgicos) ---")
    surgical_labels = ['Primary surgical', 'Secondary surgical']
    surgical_indices = df_processed[df_processed['Management'].isin(surgical_labels)].index
    
    X_surgical = features_encoded.loc[surgical_indices]
    y_surgical_severity = y2_severity[surgical_indices]

    model_severity = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    cv_scores_severity = cross_val_score(model_severity, X_surgical, y_surgical_severity, cv=5, scoring='accuracy')
    print(f"Acurácia Média (Gravidade): {cv_scores_severity.mean():.4f}")
    model_severity.fit(X_surgical, y_surgical_severity)

    # --- 4. Sistema de Apoio Inteligente (Simulação) ---
    def sistema_de_apoio_medico(dados_paciente):
        print("\n--- Análise do Paciente pelo Sistema Inteligente ---")
        dados_paciente_encoded = pd.get_dummies(dados_paciente)
        dados_paciente_aligned, _ = dados_paciente_encoded.align(features_encoded, axis=1, fill_value=0)

        pred_management_code = model_management.predict(dados_paciente_aligned)[0]
        pred_management_label = le_management.inverse_transform([pred_management_code])[0]
        print(f"🩺 **Previsão de Manejo:** {pred_management_label}")

        if pred_management_label in surgical_labels:
            pred_severity_code = model_severity.predict(dados_paciente_aligned)[0]
            pred_severity_label = le_severity.inverse_transform([pred_severity_code])[0]
            print(f"🔬 **Previsão de Gravidade:** {pred_severity_label}")
        else:
            print("Não é aplicável prever a gravidade, pois o manejo sugerido não é cirúrgico.")

    paciente_exemplo = features.head(1)
    sistema_de_apoio_medico(paciente_exemplo)

# Garante que o código só rode quando o script for executado diretamente
if __name__ == '__main__':
    main()
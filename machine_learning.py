import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
import plotly.express as px
import warnings
import pickle
from io import BytesIO

warnings.filterwarnings('ignore')

def analyze_dataset(df, target_variable):
    """Analyze dataset and provide guidance on problem type and preprocessing needs."""
    analysis = {
        'recommended_problem_type': None,
        'categorical_columns': [],
        'numeric_columns': [],
        'date_columns': [],
        'recommended_preprocessing': [],
        'warnings': [],
        'target_unique_count': None,
        'missing_values': {}
    }
    
    # Analyze target variable
    target_series = df[target_variable]
    unique_count = target_series.nunique()
    analysis['target_unique_count'] = unique_count
    
    # Detect target type
    if pd.api.types.is_numeric_dtype(target_series):
        if unique_count <= 10:  # Arbitrary threshold
            analysis['recommended_problem_type'] = 'Classification'
            analysis['warnings'].append(
                f"Target variable has {unique_count} unique numeric values. "
                "Treating this as a classification problem, but consider if regression might be more appropriate."
            )
        else:
            analysis['recommended_problem_type'] = 'Regression'
    else:
        analysis['recommended_problem_type'] = 'Classification'
    
    # Analyze features
    for column in df.columns:
        if column != target_variable:
            # Check for missing values
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                analysis['missing_values'][column] = missing_count
            
            # Detect column type
            if pd.api.types.is_numeric_dtype(df[column]):
                if df[column].nunique() <= 10:
                    analysis['categorical_columns'].append(column)
                else:
                    analysis['numeric_columns'].append(column)
            elif pd.api.types.is_datetime64_dtype(df[column]):
                analysis['date_columns'].append(column)
            else:
                analysis['categorical_columns'].append(column)
    
    # Generate preprocessing recommendations
    if analysis['missing_values']:
        analysis['recommended_preprocessing'].append(
            "Missing values detected. These will be handled automatically using median/mode imputation."
        )
    
    if analysis['categorical_columns']:
        if analysis['recommended_problem_type'] == 'Regression':
            analysis['recommended_preprocessing'].append(
                "Categorical features will be automatically one-hot encoded for regression."
            )
        else:
            analysis['recommended_preprocessing'].append(
                "Categorical features will be label encoded for classification."
            )
    
    if analysis['date_columns']:
        analysis['recommended_preprocessing'].append(
            "Date columns will be automatically converted to numeric features."
        )
    
    return analysis

def show():
    """Render the machine learning page with improved workflow and guardrails."""
    st.title("Machine Learning")

    if not st.session_state.dataframes:
        st.warning("Please upload data in the Data Upload section first.")
        return

    # Step 1: Dataset Selection
    st.header("1️⃣ Dataset Selection and Analysis")
    selected_file = st.selectbox(
        "Choose your dataset",
        list(st.session_state.dataframes.keys())
    )
    df = st.session_state.dataframes[selected_file].copy()
    
    # Dataset Overview
    with st.expander("📊 Dataset Overview", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
        
        st.markdown("#### Column Analysis")
        col_info = pd.DataFrame({
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique(),
            'Sample Values': [str(df[col].unique()[:3].tolist()) for col in df.columns]
        })
        st.dataframe(col_info)

    # Step 2: Target Selection
    st.header("2️⃣ Target Variable Selection")
    target_variable = st.selectbox(
        "Select the target variable",
        df.columns,
        help="Choose the column you want to predict"
    )
    
    # Analyze dataset
    analysis = analyze_dataset(df, target_variable)
    
    # Display Analysis Results
    st.header("3️⃣ Data Analysis Results")
    
    # Show warnings and recommendations in a clear format
    if analysis['warnings']:
        with st.expander("⚠️ Important Considerations", expanded=True):
            for warning in analysis['warnings']:
                st.warning(warning)
    
    with st.expander("🔍 Analysis Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Feature Types Found:")
            st.write(f"- Numeric Features: {len(analysis['numeric_columns'])}")
            st.write(f"- Categorical Features: {len(analysis['categorical_columns'])}")
            st.write(f"- Date Features: {len(analysis['date_columns'])}")
        
        with col2:
            st.write("Recommended Preprocessing:")
            for rec in analysis['recommended_preprocessing']:
                st.write(f"- {rec}")
    
    # Step 3: Model Configuration
    st.header("4️⃣ Model Configuration")
    
    # Problem Type Selection
    problem_type = st.radio(
        "Select problem type",
        ["Classification", "Regression"],
        index=0 if analysis['recommended_problem_type'] == 'Classification' else 1,
        help=(
            "Classification: Predict categories/classes (e.g., Yes/No, Red/Blue/Green)\n"
            "Regression: Predict continuous values (e.g., price, temperature)"
        )
    )
    
    if problem_type != analysis['recommended_problem_type']:
        st.warning(
            f"Based on the data analysis, {analysis['recommended_problem_type']} "
            f"might be more appropriate. Are you sure you want to use {problem_type}?"
        )
    
    # Feature Selection
    st.subheader("Feature Selection")
    
    # Organize features by type
    feature_types = {
        "Numeric Features": [col for col in analysis['numeric_columns'] if col != target_variable],
        "Categorical Features": analysis['categorical_columns'],
        "Date Features": analysis['date_columns']
    }
    
    selected_features = []
    
    # Create columns for different feature types
    cols = st.columns(len(feature_types))
    for i, (feature_type, features) in enumerate(feature_types.items()):
        if features:  # Only show if there are features of this type
            with cols[i]:
                st.write(f"📋 {feature_type}")
                selected = st.multiselect(
                    f"Select {feature_type.lower()}",
                    options=features,
                    default=features,
                    key=f"select_{feature_type}"
                )
                selected_features.extend(selected)
    
    if not selected_features:
        st.error("❌ Please select at least one feature for training.")
        st.stop()
    
    # Model Selection
    st.subheader("Model Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        if problem_type == "Classification":
            model_options = {
                "Random Forest": {
                    "model": RandomForestClassifier(random_state=42),
                    "description": "Good for complex relationships, handles non-linear patterns well"
                },
                "XGBoost": {
                    "model": xgb.XGBClassifier(random_state=42),
                    "description": "High performance, good with imbalanced data"
                },
                "Logistic Regression": {
                    "model": LogisticRegression(random_state=42),
                    "description": "Simple, interpretable, good for linear relationships"
                }
            }
        else:
            model_options = {
                "Random Forest": {
                    "model": RandomForestRegressor(random_state=42),
                    "description": "Good for complex relationships, handles non-linear patterns well"
                },
                "XGBoost": {
                    "model": xgb.XGBRegressor(random_state=42),
                    "description": "High performance, good with continuous data"
                },
                "Linear Regression": {
                    "model": LinearRegression(),
                    "description": "Simple, interpretable, good for linear relationships"
                }
            }
        
        selected_model_name = st.selectbox(
            "Select model type",
            list(model_options.keys())
        )
        
        st.info(f"ℹ️ {model_options[selected_model_name]['description']}")
        
        model = model_options[selected_model_name]['model']

    with col2:
        test_size = st.slider(
            "Test set size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Percentage of data to use for testing the model"
        )
        
        cv_folds = st.slider(
            "Cross-validation folds",
            min_value=2,
            max_value=10,
            value=5,
            help="Number of folds for cross-validation"
        )

    # Training Button
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                # Prepare data
                X = df[selected_features].copy()
                y = df[target_variable].copy()
                
                # Store preprocessing info
                preprocessing_info = {
                    'problem_type': problem_type,
                    'target_variable': target_variable,
                    'numeric_features': analysis['numeric_columns'],
                    'categorical_features': analysis['categorical_columns'],
                    'date_features': analysis['date_columns'],
                    'feature_names': selected_features,
                    'categorical_mappings': {},
                    'column_encoders': {},
                    'target_encoder': None,
                    'scaler': None,
                    'encoded_feature_names': None
                }
                
                # Handle missing values
                for column in X.columns:
                    if X[column].isnull().any():
                        if pd.api.types.is_numeric_dtype(X[column]):
                            fill_value = X[column].median()
                            fill_method = "median"
                        else:
                            fill_value = X[column].mode()[0]
                            fill_method = "mode"
                        X[column] = X[column].fillna(fill_value)
                        preprocessing_info[f'{column}_fill_value'] = fill_value
                        preprocessing_info[f'{column}_fill_method'] = fill_method
                
                # Handle categorical variables
                if problem_type == "Regression":
                    # For regression, use one-hot encoding
                    categorical_cols = [col for col in selected_features 
                                     if col in analysis['categorical_columns']]
                    if categorical_cols:
                        # Store original categories for each categorical column
                        for col in categorical_cols:
                            preprocessing_info['categorical_mappings'][col] = list(X[col].unique())
                        
                        # Perform one-hot encoding
                        X = pd.get_dummies(X, columns=categorical_cols)
                        preprocessing_info['encoded_feature_names'] = list(X.columns)
                else:
                    # For classification, use label encoding
                    for col in selected_features:
                        if col in analysis['categorical_columns']:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                            preprocessing_info['column_encoders'][col] = pickle.dumps(le)
                            preprocessing_info['categorical_mappings'][col] = list(le.classes_)
                
                # Scale numeric features
                numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
                if numeric_cols:
                    scaler = StandardScaler()
                    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
                    preprocessing_info['scaler'] = pickle.dumps(scaler)
                
                # Handle target variable for classification
                if problem_type == "Classification":
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y)
                    preprocessing_info['target_encoder'] = pickle.dumps(le_target)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=42,
                    stratify=y if problem_type == "Classification" else None
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Generate predictions
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X, y, cv=cv_folds)
                
                # Store model and preprocessing info in session state
                st.session_state.trained_model = pickle.dumps(model)
                st.session_state.preprocessing = preprocessing_info
                st.session_state.training_data_structure = {
                    'columns': list(X.columns),
                    'dtypes': {str(k): str(v) for k, v in X.dtypes.to_dict().items()}
                }
                
                # Display results
                st.header("5️⃣ Model Performance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Test Set Metrics")
                    if problem_type == "Classification":
                        metrics = {
                            "Accuracy": accuracy_score(y_test, y_pred),
                            "Precision": precision_score(y_test, y_pred, average='weighted'),
                            "Recall": recall_score(y_test, y_pred, average='weighted'),
                            "F1 Score": f1_score(y_test, y_pred, average='weighted')
                        }
                        
                        # Confusion Matrix
                        fig = px.imshow(
                            confusion_matrix(y_test, y_pred),
                            labels=dict(x="Predicted", y="Actual"),
                            title="Confusion Matrix"
                        )
                        st.plotly_chart(fig)
                    else:
                        metrics = {
                            "R² Score": r2_score(y_test, y_pred),
                            "MAE": mean_absolute_error(y_test, y_pred),
                            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                        }
                        
                        # Predicted vs Actual Plot
                        fig = px.scatter(
                            x=y_test, y=y_pred,
                            labels={"x": "Actual", "y": "Predicted"},
                            title="Predicted vs Actual Values"
                        )
                        fig.add_shape(
                            type='line',
                            x0=min(y_test), y0=min(y_test),
                            x1=max(y_test), y1=max(y_test),
                            line=dict(color='red', dash='dash')
                        )
                        st.plotly_chart(fig)
                    
                    # Display metrics
                    for metric_name, value in metrics.items():
                        st.metric(metric_name, f"{value:.4f}")

                with col2:
                    st.subheader("Cross-validation Results")
                    st.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
                    st.metric("CV Score Std", f"{cv_scores.std():.4f}")
                    
                    # Feature Importance
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("Feature Importance")
                        importances = pd.DataFrame({
                            'feature': X.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        fig = px.bar(
                            importances,
                            x='importance',
                            y='feature',
                            title='Feature Importance',
                            orientation='h'
                        )
                        st.plotly_chart(fig)
                
                st.success("✅ Model trained successfully! You can now make predictions below.")
                
                # Add model download capability
                if st.session_state.trained_model is not None:
                    # Create a buffer for the model and preprocessing info
                    model_buffer = BytesIO()
                    model_data = {
                        'model': st.session_state.trained_model,
                        'preprocessing': preprocessing_info,
                        'training_structure': st.session_state.training_data_structure
                    }
                    pickle.dump(model_data, model_buffer)
                    model_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Trained Model",
                        data=model_buffer.getvalue(),
                        file_name=f"{selected_model_name.lower().replace(' ', '_')}_model.pkl",
                        mime="application/octet-stream",
                        help="Download the trained model with all preprocessing information"
                    )

                # Prediction Interface
                st.header("6️⃣ Make Predictions")
                
                tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
                
                with tab1:
                    st.write("Enter values for prediction:")
                    
                    with st.form("single_prediction_form"):
                        # Initialize dictionary for input data
                        input_data = {}
                        
                        if problem_type == "Regression":
                            # For regression models, handle encoded columns
                            encoded_columns = preprocessing_info['encoded_feature_names'] or X.columns
                            
                            # Initialize all encoded columns with zeros
                            for col in encoded_columns:
                                input_data[col] = 0.0
                            
                            # Create two columns for input fields
                            col1, col2 = st.columns(2)
                            
                            # Handle numeric features
                            numeric_features = [col for col in encoded_columns 
                                             if not any(col.startswith(f"{cat_col}_") 
                                                      for cat_col in preprocessing_info['categorical_mappings'])]
                            
                            for i, feature in enumerate(numeric_features):
                                with col1 if i < len(numeric_features)/2 else col2:
                                    input_data[feature] = st.number_input(
                                        f"Enter {feature}",
                                        value=0.0,
                                        help=f"Enter a numeric value for {feature}"
                                    )
                            
                            # Handle categorical features
                            for i, (original_col, categories) in enumerate(preprocessing_info['categorical_mappings'].items()):
                                with col1 if i < len(preprocessing_info['categorical_mappings'])/2 else col2:
                                    selected_category = st.selectbox(
                                        f"Select {original_col}",
                                        options=categories,
                                        help=f"Choose a value for {original_col}"
                                    )
                                    
                                    # Set one-hot encoded values
                                    for category in categories:
                                        col_name = f"{original_col}_{category}"
                                        if col_name in encoded_columns:
                                            input_data[col_name] = 1.0 if category == selected_category else 0.0
                        
                        else:  # Classification
                            # Handle features directly
                            col1, col2 = st.columns(2)
                            features = preprocessing_info['feature_names']
                            
                            for i, feature in enumerate(features):
                                with col1 if i < len(features)/2 else col2:
                                    if feature in preprocessing_info['categorical_mappings']:
                                        # Categorical feature
                                        categories = preprocessing_info['categorical_mappings'][feature]
                                        selected_category = st.selectbox(
                                            f"Select {feature}",
                                            options=categories,
                                            help=f"Choose a value for {feature}"
                                        )
                                        
                                        le = pickle.loads(preprocessing_info['column_encoders'][feature])
                                        input_data[feature] = le.transform([selected_category])[0]
                                    else:
                                        # Numeric feature
                                        input_data[feature] = st.number_input(
                                            f"Enter {feature}",
                                            value=0.0,
                                            help=f"Enter a numeric value for {feature}"
                                        )
                        
                        predict_button = st.form_submit_button("Make Prediction")
                    
                    if predict_button:
                        try:
                            # Load model from session state
                            model = pickle.loads(st.session_state.trained_model)
                            
                            # Create DataFrame with exact structure
                            input_df = pd.DataFrame([input_data], 
                                                  columns=preprocessing_info['encoded_feature_names'] or X.columns)
                            
                            # Scale numeric features if necessary
                            if preprocessing_info.get('scaler') is not None:
                                numeric_cols = [col for col in input_df.columns 
                                              if col in st.session_state.training_data_structure['columns']]
                                if numeric_cols:
                                    scaler = pickle.loads(preprocessing_info['scaler'])
                                    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
                            
                            # Make prediction
                            prediction = model.predict(input_df)
                            
                            st.subheader("Prediction Result")
                            
                            if problem_type == "Classification":
                                # Transform prediction back to original class
                                le_target = pickle.loads(preprocessing_info['target_encoder'])
                                predicted_class = le_target.inverse_transform(prediction)[0]
                                st.success(f"Predicted Class: {predicted_class}")
                                
                                # Show probabilities if available
                                if hasattr(model, 'predict_proba'):
                                    probabilities = model.predict_proba(input_df)
                                    prob_df = pd.DataFrame(
                                        probabilities,
                                        columns=le_target.classes_
                                    )
                                    
                                    fig = px.bar(
                                        x=prob_df.columns,
                                        y=prob_df.iloc[0],
                                        title="Class Probabilities",
                                        labels={"x": "Class", "y": "Probability"}
                                    )
                                    st.plotly_chart(fig)
                            else:
                                st.success(f"Predicted Value: {prediction[0]:.4f}")
                            
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
                            st.error("Please ensure all input values are in the correct format.")
                
                with tab2:
                    st.write("Upload a CSV file for batch predictions:")
                    
                    # Provide template download
                    template_data = {}
                    if problem_type == "Regression":
                        # For regression, provide template with original column names
                        original_features = preprocessing_info['feature_names']
                        template_data = {col: [] for col in original_features}
                    else:
                        # For classification, use feature names directly
                        template_data = {col: [] for col in preprocessing_info['feature_names']}
                    
                    template_df = pd.DataFrame(template_data)
                    
                    st.download_button(
                        "📥 Download Template CSV",
                        template_df.to_csv(index=False),
                        "prediction_template.csv",
                        "text/csv",
                        help="Download a template CSV with the required columns"
                    )
                    
                    uploaded_file = st.file_uploader(
                        "Upload your CSV file",
                        type=['csv'],
                        help="Upload a CSV file with the same columns as the template"
                    )
                    
                    if uploaded_file is not None:
                        try:
                            predict_df = pd.read_csv(uploaded_file)
                            
                            st.write("Preview of uploaded data:")
                            st.dataframe(predict_df.head())
                            
                            if st.button("Make Batch Predictions"):
                                model = pickle.loads(st.session_state.trained_model)
                                
                                with st.spinner("Processing predictions..."):
                                    try:
                                        # Process the data
                                        if problem_type == "Regression":
                                            # Handle one-hot encoding for regression
                                            categorical_cols = list(preprocessing_info['categorical_mappings'].keys())
                                            processed_df = pd.get_dummies(predict_df, columns=categorical_cols)
                                            
                                            # Ensure all columns from training are present
                                            missing_cols = set(preprocessing_info['encoded_feature_names']) - set(processed_df.columns)
                                            for col in missing_cols:
                                                processed_df[col] = 0
                                            
                                            # Reorder columns to match training data
                                            processed_df = processed_df[preprocessing_info['encoded_feature_names']]
                                        else:
                                            # Handle label encoding for classification
                                            processed_df = predict_df.copy()
                                            for col, encoder_bytes in preprocessing_info['column_encoders'].items():
                                                encoder = pickle.loads(encoder_bytes)
                                                processed_df[col] = encoder.transform(processed_df[col].astype(str))
                                        
                                        # Scale numeric features if necessary
                                        if preprocessing_info.get('scaler') is not None:
                                            scaler = pickle.loads(preprocessing_info['scaler'])
                                            numeric_cols = [col for col in processed_df.columns 
                                                          if col in st.session_state.training_data_structure['columns']]
                                            if numeric_cols:
                                                processed_df[numeric_cols] = scaler.transform(processed_df[numeric_cols])
                                        
                                        # Make predictions
                                        predictions = model.predict(processed_df)
                                        
                                        # Prepare results DataFrame
                                        results_df = predict_df.copy()
                                        if problem_type == "Classification":
                                            le_target = pickle.loads(preprocessing_info['target_encoder'])
                                            predictions = le_target.inverse_transform(predictions)
                                            results_df['Predicted_Class'] = predictions
                                            
                                            if hasattr(model, 'predict_proba'):
                                                probabilities = model.predict_proba(processed_df)
                                                for i, class_name in enumerate(le_target.classes_):
                                                    results_df[f'Probability_{class_name}'] = probabilities[:, i]
                                        else:
                                            results_df['Predicted_Value'] = predictions
                                        
                                        # Display results
                                        st.subheader("Prediction Results")
                                        st.dataframe(results_df)
                                        
                                        # Provide download option
                                        st.download_button(
                                            "📥 Download Predictions",
                                            results_df.to_csv(index=False),
                                            "predictions.csv",
                                            "text/csv",
                                            help="Download the predictions as a CSV file"
                                        )
                                        
                                    except Exception as e:
                                        st.error(f"Error during batch prediction: {str(e)}")
                                        st.error("Please ensure your data matches the required format.")
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
                            st.error("Please ensure your CSV file is properly formatted.")

            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")
                st.error("Please check your data and selected options.")
                st.exception(e)
from preprocess import load_data, get_features_and_target, create_preprocessor

# load the dataset
df = load_data()
print("Dataset shape: ", df.shape)

# separate features and target
X, y = get_features_and_target(df)

print("\nSelected feature columns:")
print(X.columns.tolist())

# preprocessing pipeline
preprocessor = create_preprocessor(X)

# fit and transform the data
X_processed = preprocessor.fit_transform(X)

print("\nOriginal feature shape: ", X.shape)
print("\nProcessed feature shape: ", X_processed.shape)

# show detected column types

categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
print("\nCategorical features: ", categorical_features)

numerical_features = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

print("\nNumerical Features: ", numerical_features)
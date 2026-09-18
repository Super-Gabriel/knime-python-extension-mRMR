import knime.extension as knext
import fast_mRMR

@knext.node(
    name="mRMR Feature Selector",
    node_type=knext.NodeType.LEARNER,
    icon_path="../icons/icon.png",
    category="/community/mRMR",
)
@knext.input_table(
    name="Input Table",
    description="Table containing the X features and the target Y column",
)
@knext.output_table(
    name="Selected Features",
    description="Table containing the selected feature columns",
)
class MRMRNode:
    """
    Nodo que aplica mRMR para seleccionar las k características más relevantes
    de la tabla X, usando la información de la tabla Y.
    """
    y_column = knext.ColumnParameter(
        label="Target column (Y)",
        description="Select the column to use as the target variable",
    )

    k_features = knext.IntParameter(
        label="Number of features",
        description="Number of features to select with mRMR",
        default_value=5,
        min_value=1,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context, input_table):
        import pandas as pd
        from sklearn.preprocessing import KBinsDiscretizer
        df = input_table.to_pandas()

        y = df[self.y_column]
        X = df.drop(columns=[self.y_column])
        
        k = self.k_features
        
        discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
        X_disc = discretizer.fit_transform(X)
        X_disc_df = pd.DataFrame(X_disc, columns=X.columns)
        
        selected_columns = fast_mRMR.fast_mrmr(X_disc_df, y, k)
        
        X_selected = X[selected_columns]
        
        return knext.Table.from_pandas(X_selected)
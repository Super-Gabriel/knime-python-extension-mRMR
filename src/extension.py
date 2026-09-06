import knime.extension as knext
import pandas as pd
import fast_mRMR

@knext.node(
    name="mRMR Feature Selector",
    node_type=knext.NodeType.LEARNER,
    icon_path="../icons/icon.png",
    category="/community/mRMR",
)
@knext.input_table(name="X Table", description="Tabla de características (X)")
@knext.input_table(name="Y Table", description="Tabla de etiquetas (Y)")
@knext.output_table(name="Selected Features", description="Tabla X con solo las columnas seleccionadas")
class MRMRNode:
    """
    Nodo que aplica mRMR para seleccionar las k características más relevantes
    de la tabla X, usando la información de la tabla Y.
    """

    # Parámetro en el diálogo de configuración
    k_features = knext.IntParameter(
        label="Número de características",
        description="Cantidad de features a seleccionar con mRMR",
        default_value=5,
        min_value=1,
    )

    def configure(self, configure_context, input_schema_1, input_schema_2):
        # Devolvemos el mismo esquema de X (primer puerto) sin cambios
        # (mismo número de columnas, solo filtramos en ejecución)
        # return input_schema_1
        return None

    def execute(self, exec_context, input_1, input_2):
        import pandas as pd
        from sklearn.preprocessing import KBinsDiscretizer
        
        # Convertir las tablas KNIME a DataFrames
        X = input_1.to_pandas()          # DataFrame con todas las características
        Y_df = input_2.to_pandas()       # DataFrame con la(s) columna(s) de etiquetas
        # Extraer la primera columna de Y como Serie (vector 1D)
        y = Y_df.iloc[:, 0]
        
        k = self.k_features
        
        # ---- Discretización (misma que en tu proyecto) ----
        discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
        X_disc = discretizer.fit_transform(X)
        X_disc_df = pd.DataFrame(X_disc, columns=X.columns)
        
        # ---- Selección con fast_mRMR ----
        selected_columns = fast_mRMR.fast_mrmr(X_disc_df, y, k)
        
        # Filtrar las columnas originales (no las discretizadas)
        X_selected = X[selected_columns]
        
        return knext.Table.from_pandas(X_selected)
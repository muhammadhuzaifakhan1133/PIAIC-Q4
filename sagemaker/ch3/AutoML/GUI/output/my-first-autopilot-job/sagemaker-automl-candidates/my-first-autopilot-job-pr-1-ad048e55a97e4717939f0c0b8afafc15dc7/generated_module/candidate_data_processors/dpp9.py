from datetime import datetime as dt
from sagemaker_sklearn_extension.externals import Header
from sagemaker_sklearn_extension.feature_extraction.date_time import DateTimeVectorizer
from sagemaker_sklearn_extension.impute import RobustImputer
from sagemaker_sklearn_extension.preprocessing import RobustLabelEncoder
from sagemaker_sklearn_extension.preprocessing import RobustOrdinalEncoder
from sagemaker_sklearn_extension.preprocessing import RobustStandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Given a list of column names and target column name, Header can return the index
# for given column name
HEADER = Header(
    column_names=[
        'y', 'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
        'cons.conf.idx', 'euribor3m', 'nr.employed'
    ],
    target_column_name='y'
)


def build_feature_transform():
    """ Returns the model definition representing feature processing."""

    # These features contain a relatively small number of unique items. If this list is
    # modified, the value of the `num_categorical_features` hyperparameter must be adjusted to
    # reflect the number of features in the modified list. To make this change please find this
    # pipeline in the "Candidate Pipelines" section of the
    # SageMakerAutopilotCandidateDefinitionNotebook and change the value of
    # `num_categorical_features` in the `candidate_specific_static_hyperameters` dictionary.

    categorical = HEADER.as_feature_indices(
        [
            'job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'poutcome'
        ]
    )

    # These features can be parsed as numeric.

    numeric = HEADER.as_feature_indices(
        [
            'age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
            'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
        ]
    )

    # These features can be parsed as date or time.

    datetime = HEADER.as_feature_indices(['month', 'day_of_week'])

    categorical_processors = Pipeline(
        steps=[
            (
                'robustordinalencoder',
                RobustOrdinalEncoder(threshold='auto', max_categories=100)
            )
        ]
    )

    numeric_processors = Pipeline(
        steps=[
            ('robustimputer',
             RobustImputer()), ('robuststandardscaler', RobustStandardScaler())
        ]
    )

    datetime_processors = Pipeline(
        steps=[
            (
                'datetimevectorizer',
                DateTimeVectorizer(
                    mode='cyclic',
                    default_datetime=dt(year=1970, month=1, day=1)
                )
            ), ('robustimputer1', RobustImputer())
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ('categorical_processing', categorical_processors,
             categorical), ('numeric_processing', numeric_processors, numeric),
            ('datetime_processing', datetime_processors, datetime)
        ]
    )

    return Pipeline(steps=[('column_transformer', column_transformer)])


def build_label_transform():
    """Returns the model definition representing feature processing."""

    return RobustLabelEncoder(
        labels=['no'], fill_label_value='yes', include_unseen_class=True
    )

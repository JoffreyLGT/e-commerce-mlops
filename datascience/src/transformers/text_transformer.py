# type: ignore
# ruff: noqa
"""All classes and functions used to preprocess text data."""

import logging
import pickle
import re
from html.parser import HTMLParser

import pandas as pd
import tensorflow as tf
from nltk.stem.snowball import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.pipeline import Pipeline

from src.utilities.dataset_utils import convert_sparse_matrix_to_sparse_tensor

logger = logging.getLogger(__file__)


class TextPreprocess:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def fit(self, data):
        return self.pipeline.fit(data)

    def fit_transform(self, data) -> tf.SparseTensor:
        out = self.pipeline.fit_transform(data)
        return convert_sparse_matrix_to_sparse_tensor(out)

    def transform(self, data) -> tf.SparseTensor:
        out = self.pipeline.transform(data)
        return convert_sparse_matrix_to_sparse_tensor(out)

    def get_voc(self):
        return self.pipeline.get_voc()

    def save_voc(self, prefix_filename):
        voc = self.get_voc()
        file_name = f"{prefix_filename}_{self.pipeline.name}.pkl"
        with open(file_name, "wb") as fp:
            pickle.dump(voc, fp)
        logger.info(f"TextPreprocess.save_voc {file_name}")
        return file_name


class _RakutenHTMLParser(HTMLParser):
    """Parse the textand return the content without HTML tag or encoding."""

    def __init__(self):
        self.allcontent = ""
        super().__init__()

    def handle_data(self, data):
        self.allcontent += data + " "

    def get_all_content(self):
        return self.allcontent.strip()


class HTMLRemover(BaseEstimator, TransformerMixin):
    """Transformer removing HTML tags and decoding HTML special characters."""

    def _parseValue(self, value):
        if type(value) != str:
            return value
        parser = _RakutenHTMLParser()
        parser.feed(value)
        return parser.get_all_content()

    def _parseColumn(self, column):
        return [self._parseValue(value) for value in column]

    def fit(self, X, y=None):
        # Do nothing, mandatory function for when a model is provided to the pipeline.
        return self

    def transform(self, X):
        if type(X) == pd.DataFrame:
            return X.apply(lambda column: self._parseColumn(column))

        return X.apply(lambda column: self._parseValue(column))


class NumRemover(BaseEstimator, TransformerMixin):
    """Remove all number from strings."""

    def _parse_value(self, value):
        if isinstance(type(value), int):
            return value
        return re.sub("\s?([0-9]+)\s?", " ", value)

    def _parse_column(self, column):
        return [self._parse_value(value) for value in column]

    def fit(self, X, y=None):
        # Do nothing, mandatory function for when a model is provided to the pipeline.
        return self

    def transform(self, X):
        if type(X) == pd.DataFrame:
            return X.apply(lambda column: self._parse_column(column))

        return X.apply(lambda column: self._parse_value(column))


class StemmedCountVectorizer(CountVectorizer):
    fr_stemmer = SnowballStemmer("french")

    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: (
            StemmedCountVectorizer.fr_stemmer.stem(w) for w in analyzer(doc)
        )


class StemmedTfidfVectorizer(TfidfVectorizer):
    fr_stemmer = SnowballStemmer("french")

    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: (
            StemmedTfidfVectorizer.fr_stemmer.stem(w) for w in analyzer(doc)
        )


class TfidfStemming(Pipeline):
    def __init__(self) -> None:
        self.name = "TfidfStemming"
        steps = [
            ("remove_html", HTMLRemover()),
            ("remove_num", NumRemover()),
            ("tfidStem", StemmedTfidfVectorizer()),
        ]
        Pipeline.__init__(self, steps)

    def get_voc(self):
        return self.steps[2][1].vocabulary_

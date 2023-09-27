"""All classes and functions used to preprocess text data."""

import json
import logging
import re
from collections.abc import Iterable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Self

import numpy as np
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
    """Manage text preprocessing to clean and transform text for our MLP model."""

    def __init__(
        self,
        vocabulary: dict[str, int] | None = None,
        idfs: Iterable[float] | None = None,
    ) -> None:
        """Initialize instance attributes.

        Args:
            vocabulary: used to restore Tfidf vectorizer.
            idfs: used to restore Tfidf vectorize.
        """
        self.pipeline = TfidfStemming()
        if vocabulary is not None and idfs is not None:
            self.pipeline.update_voc_idfs(vocabulary, idfs)

    def fit(self, data: Iterable[str]):  # type: ignore
        """Analyze the data to prepare the transformation.

        Args:
            data: contains the text to analyze.
        """
        return self.pipeline.fit(data)

    def fit_transform(self, data: Iterable[str]) -> tf.SparseTensor:
        """Analyze and transform data.

        Args:
            data: strings to transform.

        Returns:
            Transformed data as SparseTensor.
        """
        out = self.pipeline.fit_transform(data)
        return convert_sparse_matrix_to_sparse_tensor(out)

    def transform(self, data: Iterable[str]) -> tf.SparseTensor:
        """Transform data into a sparse tensor.

        Args:
            data: strings to transform.

        Returns:
            Transformed data as SparseTensor.
        """
        out = self.pipeline.transform(data)
        return convert_sparse_matrix_to_sparse_tensor(out)

    def get_voc_idfs(self) -> tuple[dict[str, int], list[float]]:
        """Returns vocabulary and idfs from tfidf vecorizer."""
        return self.pipeline.get_voc_idfs()

    def save_voc(
        self, vocabulary_file_path: str | Path, idfs_file_path: str | Path
    ) -> None:
        """Save vocabulary and idfs into json files.

        Args:
            vocabulary_file_path: destination of vocabulary json file.
            idfs_file_path: destination of idfs json file.
        """
        voc, idfx = self.get_voc_idfs()
        with Path(vocabulary_file_path).open("w") as file:
            json.dump(voc, file)
        with Path(idfs_file_path).open("w") as file:
            json.dump(idfx, file)


class _RakutenHTMLParser(HTMLParser):
    """Parse the text and return the content without HTML tag or encoding."""

    def __init__(self) -> None:
        """Initialize attributes."""
        self.allcontent = ""
        super().__init__()

    def handle_data(self, data: str) -> None:
        """Load content into an attribute.

        Args:
            data: content to load.
        """
        self.allcontent += data + " "

    def get_all_content(self) -> str:
        """Get all content stripped of any HTML tag."""
        return self.allcontent.strip()


class HTMLRemover(BaseEstimator, TransformerMixin):  # type: ignore
    """Transformer removing HTML tags and decoding HTML special characters."""

    def _parse_value(self, value: str) -> str:
        """Parse provided value and return it without HTML tags."""
        if isinstance(value, str):
            return value
        parser = _RakutenHTMLParser()
        parser.feed(value)
        return parser.get_all_content()

    def _parse_column(self, column: Iterable[str]) -> Iterable[str]:
        return [self._parse_value(value) for value in column]

    def fit(self, _x: Any, _y: Any = None) -> Self:  # pyright: ignore
        """Do nothing, mandatory function so it can be used into the pipeline."""
        return self

    def transform(
        self, X: pd.DataFrame | pd.Series  # type: ignore
    ) -> Iterable[str] | str:
        """Remove all HTML tag and decode HTML special character from data.

        Args:
            X: data to clean.

        Returns:
            Cleaned data.
        """
        if isinstance(X, pd.DataFrame):
            return X.apply(lambda column: self._parse_column(column))  # type: ignore

        return X.apply(lambda column: self._parse_value(column))


class NumRemover(BaseEstimator, TransformerMixin):  # type:ignore
    """Remove all number from strings."""

    def _parse_value(self, value: str) -> str:
        if isinstance(type(value), int):
            return value
        return re.sub("\\s?([0-9]+)\\s?", " ", value)

    def _parse_column(self, column: Iterable[str]) -> Iterable[str]:
        return [self._parse_value(value) for value in column]

    def fit(self, _x: Any, _y: Any = None) -> Self:  # pyright: ignore
        """Do nothing, mandatory function so it can be used into the pipeline."""
        return self

    def transform(
        self, X: pd.DataFrame | pd.Series | str  # type:ignore
    ) -> Iterable[str] | str:
        """Transform data by removind the numbers.

        Args:
            X: data to transform.

        Returns:
            Data without numbers.
        """
        if type(X) == pd.DataFrame:
            return X.apply(lambda column: self._parse_column(column))  # type: ignore
        if type(X) == pd.Series:
            return X.apply(lambda column: self._parse_value(column))

        return self._parse_value(X)  # type: ignore


class StemmedCountVectorizer(CountVectorizer):  # type: ignore  # noqa: D101
    fr_stemmer = SnowballStemmer("french")

    def build_analyzer(self):  # type: ignore  # noqa: D102
        analyzer = super().build_analyzer()
        return lambda doc: (
            StemmedCountVectorizer.fr_stemmer.stem(w) for w in analyzer(doc)
        )


class StemmedTfidfVectorizer(TfidfVectorizer):  # type: ignore  # noqa: D101
    fr_stemmer = SnowballStemmer("french")

    def update_voc_idfs(
        self, vocabulary: dict[str, int] | None, idfs: Iterable[float] | None
    ) -> None:
        """Update vocabulary and idfs to not have to fit again.

        Args:
            vocabulary: to update.
            idfs: to update.
        """
        if idfs is not None:
            self.idf_ = idfs
        if vocabulary is not None:
            self.vocabulary_ = vocabulary

    def build_analyzer(self):  # type:ignore  # noqa: D102
        analyzer = super().build_analyzer()
        return lambda doc: (
            StemmedTfidfVectorizer.fr_stemmer.stem(w) for w in analyzer(doc)
        )


class TfidfStemming(Pipeline):  # type:ignore
    """Pipeline to do stemming on the text and convert it to a vector with tfidf."""

    def __init__(
        self,
    ) -> None:
        """Initialize object attributes, expecially the pipeline steps."""
        self.name = "TfidfStemming"
        steps = [
            ("remove_html", HTMLRemover()),
            ("remove_num", NumRemover()),
            ("tfidStem", StemmedTfidfVectorizer()),
        ]
        Pipeline.__init__(self, steps)

    def update_voc_idfs(
        self, vocabulary: dict[str, int], idfs: Iterable[float]
    ) -> None:
        """Update vocabulary and idfs of Tfidf vectorizer.

        Args:
            vocabulary: mapping of terms to feature indices.
            idfs: inverse document frequency vectore.
        """
        self.steps[2][1].update_voc_idfs(vocabulary, np.array(idfs))

    def get_voc_idfs(self) -> tuple[dict[str, int], list[float]]:
        """Return vocabulary and idfs."""
        return self.steps[2][1].vocabulary_, self.steps[2][1].idf_.tolist()

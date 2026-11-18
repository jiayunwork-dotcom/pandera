"""Handle schema errors."""

from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Union

from pandera.api.checks import Check
from pandera.config import ValidationDepth, get_config_context
from pandera.errors import SchemaError, SchemaErrorReason
from pandera.validation_depth import ValidationScope, validation_type


class ErrorCategory(Enum):
    """Error category codes"""

    DATA = "data-failures"
    SCHEMA = "schema-failures"
    DTYPE_COERCION = "dtype-coercion-failures"


class ErrorHandler:
    """Handler for Schema & Data level errors during validation."""

    def __init__(
        self, lazy: bool = True, validation_context: Any = None
    ) -> None:
        """Initialize ErrorHandler.

        :param lazy: if True, lazily evaluates all checks before raising the exception.
        Defaults to True.
        :param validation_context: per-validation bookkeeping object for
            conditional (``where``) schemas; ``None`` for ordinary schemas.
        """
        self._lazy = lazy
        self._collected_errors: List[Dict[str, Any]] = []
        self._schema_errors: List[SchemaError] = []
        self._summarized_errors = defaultdict(lambda: defaultdict(list))  # type: ignore
        self.validation_context = validation_context
        # Structural errors, unfinished coercion/parsing and parser execution
        # failures can never be repaired by deleting rows. This flag lets the
        # final ``drop_invalid_rows`` step know that dropping must not run.
        self.has_non_row_error = False

    @property
    def lazy(self) -> bool:
        """Whether or not the schema error handler raises errors immediately."""
        return self._lazy

    def mark_non_row_error(self, schema_error=None) -> None:
        """Record that an error not attributable to concrete rows occurred.

        :param schema_error: when provided, the error is tagged directly so
            the classification survives re-collection by a parent handler.
        """
        self.has_non_row_error = True
        if schema_error is not None:
            schema_error.non_row_error = True
        if self.validation_context is not None:
            self.validation_context.mark_unlocatable_error()

    def collect_error(
        self,
        error_type: ErrorCategory,
        reason_code: SchemaErrorReason,
        schema_error: SchemaError,
        original_exc: Union[BaseException, None] = None,
    ):
        """Collect schema error, raising exception if lazy is False.

        :param error_type: type of error
        :param reason_code: string representing reason for error
        :param schema_error: ``SchemaError`` object.
        """
        if not self._lazy:
            raise schema_error from original_exc

        # delete data of validated object from SchemaError object to prevent
        # storing copies of the validated DataFrame/Series for every
        # SchemaError collected.
        if hasattr(schema_error, "data"):
            del schema_error.data

        schema_error.data = None

        # structural / conversion / parser-execution failures are marked on the
        # error itself so the classification survives being re-raised and
        # re-collected across nested error handlers.
        if getattr(schema_error, "non_row_error", False):
            self.has_non_row_error = True

        # conditional validation: accumulate the exact physical rows each
        # error attributes, and flag errors that cannot be located so the
        # drop step knows not to guess. The marker also lets the error
        # formatter add the row_position column even when no individual
        # error is locatable (e.g. a missing column).
        if self.validation_context is not None:
            schema_error.conditional_validation = True
            row_positions = getattr(schema_error, "row_positions", None)
            if row_positions:
                self.validation_context.add_failure_positions(row_positions)
            else:
                self.validation_context.mark_unlocatable_error()

        self._schema_errors.append(schema_error)

        # Failure cases can be a dataframe-like object or a scalar value. Try
        # getting the number of elements in failure cases column or set to one.
        try:
            failure_cases_count = len(schema_error.failure_cases)
        except TypeError:
            if schema_error.failure_cases is None:
                failure_cases_count = 0
            else:
                failure_cases_count = 1

        self._collected_errors.append(
            {
                "type": error_type,
                "column": schema_error.schema.name,
                "check": schema_error.check,
                "reason_code": reason_code,
                "error": schema_error,
                "failure_cases_count": failure_cases_count,
            }
        )

    def collect_errors(
        self,
        schema_errors: List[SchemaError],
        original_exc: Union[BaseException, None] = None,
    ):
        """Collect schema errors from a SchemaErrors exception.

        :param reason_code: string representing reason for error.
        :param schema_error: ``SchemaError`` object.
        :param original_exc: original exception associated with the SchemaError.
        """
        for schema_error in schema_errors:
            self.collect_error(
                validation_type(schema_error.reason_code),
                schema_error.reason_code,
                schema_error,
                original_exc or schema_error,
            )

    @property
    def collected_errors(self) -> List[Dict[str, Any]]:
        """Retrieve error objects collected during lazy validation."""
        return self._collected_errors

    @collected_errors.setter
    def collected_errors(self, value: List[Dict[str, Any]]):
        """Set the list of collected errors."""
        if not isinstance(value, list):
            raise ValueError("collected_errors must be a list")
        self._collected_errors = value

    @property
    def schema_errors(self) -> List[SchemaError]:
        """Retrieve SchemaError objects collected during lazy validation."""
        return self._schema_errors

    def summarize(self, schema_name):
        """Collect schema error, raising exception if lazy is False.

        :param schema: schema object
        """

        for e in self._collected_errors:
            category = e["type"].name
            subcategory = e["reason_code"].name
            error = e["error"]

            if self.invalid_reason_code(category):
                continue

            if isinstance(error.check, Check):
                check = error.check.error or error.check.name
            else:
                check = error.check

            # Include error["failure_cases_count"] on the summary as a future
            # improvement
            self._summarized_errors[category][subcategory].append(
                {
                    "schema": schema_name,
                    "column": e["column"],
                    "check": check,
                    "error": error.__str__().replace("\n", ""),
                }
            )

        return self._summarized_errors

    def invalid_reason_code(self, category):
        """Determine if the check should be included in the error report

        :param category: Enum object
        """
        config = get_config_context()
        if config.validation_depth == ValidationDepth.SCHEMA_AND_DATA:
            return False
        elif (
            config.validation_depth == ValidationDepth.DATA_ONLY
            and category == ValidationScope.DATA.name
        ):
            return False
        elif (
            config.validation_depth == ValidationDepth.SCHEMA_ONLY
            and category == ValidationScope.SCHEMA.name
        ):
            return False

        return True

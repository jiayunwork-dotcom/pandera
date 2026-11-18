"""Backend implementation for pandas schema components."""

# pylint: disable=too-many-locals

import copy as copy_module
import traceback
from copy import deepcopy
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from pandera.api.base.error_handler import ErrorHandler
from pandera.api.pandas.types import (
    is_field,
    is_index,
    is_multiindex,
    is_table,
)
from pandera.backends.base import CoreCheckResult
from pandera.backends.pandas.array import ArraySchemaBackend
from pandera.backends.pandas.container import DataFrameSchemaBackend
from pandera.errors import (
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
from pandera.validation_depth import validation_type


# Reason codes that describe structural / conversion failures rather than
# individual bad rows. They cannot be repaired by ``drop_invalid_rows``.
_NON_ROW_REASON_CODES = frozenset(
    {
        SchemaErrorReason.DATATYPE_COERCION,
        SchemaErrorReason.WRONG_DATATYPE,
        SchemaErrorReason.WRONG_FIELD_NAME,
        SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
        SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
        SchemaErrorReason.COLUMN_NOT_ORDERED,
        SchemaErrorReason.DUPLICATE_COLUMN_LABELS,
        SchemaErrorReason.INVALID_COLUMN_NAME,
        SchemaErrorReason.ADD_MISSING_COLUMN_NO_DEFAULT,
        SchemaErrorReason.MISMATCH_INDEX,
    }
)


class ColumnBackend(ArraySchemaBackend):
    """Backend implementation for pandas dataframe columns."""

    def validate(
        self,
        check_obj: pd.DataFrame,
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
        validation_context=None,
    ) -> pd.DataFrame:
        # pylint: disable=too-many-branches
        """Validation backend implementation for pandas dataframe columns.."""
        if not inplace:
            check_obj = check_obj.copy()

        error_handler = ErrorHandler(
            lazy=lazy, validation_context=validation_context
        )

        if getattr(schema, "drop_invalid_rows", False) and not lazy:
            raise SchemaDefinitionError(
                "When drop_invalid_rows is True, lazy must be set to True."
            )

        if schema.name is None:
            raise SchemaError(
                schema,
                check_obj,
                "column name is set to None. Pass the ``name`` argument when "
                "initializing a Column object, or use the ``set_name`` "
                "method.",
                reason_code=SchemaErrorReason.INVALID_COLUMN_NAME,
            )

        def validate_column(check_obj, column_name, return_check_obj=False):
            try:
                # pylint: disable=super-with-arguments
                # make sure the schema component mutations are reverted after
                # validation
                _orig_name = schema.name
                validated_check_obj = super(ColumnBackend, self).validate(
                    check_obj,
                    schema.set_name(column_name),
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=inplace,
                    validation_context=validation_context,
                )
                # revert the schema component mutations
                schema.name = _orig_name

                if return_check_obj:
                    return validated_check_obj

            except SchemaErrors as errs:
                for err in errs.schema_errors:
                    err.column_name = column_name
                    if err.reason_code in _NON_ROW_REASON_CODES:
                        error_handler.mark_non_row_error(err)
                    error_handler.collect_error(
                        validation_type(err.reason_code), err.reason_code, err
                    )
            except SchemaError as err:
                err.column_name = column_name
                if err.reason_code in _NON_ROW_REASON_CODES:
                    error_handler.mark_non_row_error(err)
                error_handler.collect_error(
                    validation_type(err.reason_code), err.reason_code, err
                )

        column_keys_to_check = (
            self.get_regex_columns(schema, check_obj)
            if schema.regex
            else [schema.name]
        )

        for column_name in column_keys_to_check:
            if pd.notna(schema.default):
                check_obj[column_name] = check_obj[column_name].fillna(
                    schema.default
                )
            if schema.coerce:
                try:
                    check_obj[column_name] = self.coerce_dtype(
                        check_obj[column_name],
                        schema=schema,
                    )
                except SchemaErrors as exc:
                    for schema_error in exc.schema_errors:
                        schema_error.column_name = column_name
                        error_handler.mark_non_row_error(schema_error)
                    error_handler.collect_errors(exc.schema_errors)

            if is_table(check_obj[column_name]):
                for i in range(check_obj[column_name].shape[1]):
                    validated_column = validate_column(
                        check_obj[column_name].iloc[:, [i]],
                        column_name,
                        return_check_obj=True,
                    )
                    if schema.parsers:
                        check_obj[column_name] = validated_column
                continue

            # Run the column's parsers once. A failed parser or an output with
            # an illegal row layout is reported as a CHECK_ERROR attributed to
            # the actual column and parser; the bad output is discarded so it
            # can never overwrite the valid table, and the column's checks are
            # skipped (they would consume parser output that does not exist).
            # Other columns continue independently.
            parser_failure = None
            if schema.parsers:
                parsed, parser_failure = self._run_column_parsers_safely(
                    check_obj,
                    schema,
                    column_name,
                    validation_context,
                    error_handler,
                )
                if parser_failure is None:
                    if isinstance(parsed, pd.Series):
                        check_obj[column_name] = parsed
                    elif is_table(parsed) and column_name in parsed.columns:
                        check_obj[column_name] = parsed[column_name]
                    else:
                        check_obj = parsed

            if parser_failure is not None:
                # the check pass relies on parser-produced values that do not
                # exist; do not fabricate success or failure records for it.
                continue

            if schema.parsers:
                # Parsers have already run exactly once. Reassemble the table
                # (conditional checks must keep seeing every column because the
                # `where` callable may read other columns) and run only the
                # checks via a parser-free copy so parsers are not re-executed.
                check_only = self._check_only_component(schema)
                validated_column = check_only.get_backend(check_obj).validate(
                    check_obj,
                    schema=check_only.set_name(column_name),
                    head=head,
                    tail=tail,
                    sample=sample,
                    random_state=random_state,
                    lazy=lazy,
                    inplace=True,
                    validation_context=validation_context,
                )
            elif (
                getattr(schema, "drop_invalid_rows", False)
                and validation_context is None
            ):
                # replace the check_obj with the validated. In conditional mode
                # rows are dropped once, at the end of the whole schema
                # validation, using physical positions; the column backend only
                # accumulates failure positions.
                check_obj = validate_column(
                    check_obj, column_name, return_check_obj=True
                )
                validated_column = check_obj
            else:
                validated_column = validate_column(
                    check_obj,
                    column_name,
                    return_check_obj=True,
                )

            if schema.parsers:
                if isinstance(validated_column, pd.Series):
                    check_obj[column_name] = validated_column
                elif is_table(validated_column):
                    check_obj[column_name] = validated_column[column_name]

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        return check_obj

    def _run_column_parsers_safely(
        self,
        check_obj,
        schema,
        column_name,
        validation_context,
        error_handler,
    ):
        """Execute a column's parsers, turning any failure into a SchemaError.

        Returns ``(parsed_obj, failure)`` where ``failure`` is ``None`` on
        success. Every parser runs at most once. The returned ``parsed_obj`` is
        only valid when ``failure`` is ``None``.
        """
        parsed_obj = check_obj
        for parser_index, parser in enumerate(schema.parsers):
            before = parsed_obj
            try:
                result = self.run_parser(
                    parsed_obj,
                    parser,
                    parser_index,
                    column_name,
                )
            except SchemaError as err:
                return None, self._column_parser_error(
                    schema,
                    check_obj,
                    column_name,
                    parser,
                    parser_index,
                    err,
                    validation_context,
                    error_handler,
                )
            except Exception as err:  # pylint: disable=broad-except
                return None, self._column_parser_error(
                    schema,
                    check_obj,
                    column_name,
                    parser,
                    parser_index,
                    err,
                    validation_context,
                    error_handler,
                )
            parsed_obj = result.parser_output

            # verify the output keeps a legal row layout. The reference is the
            # parser's own input: a table output must keep the input table's
            # rows, a Series output must keep the rows of the Series/column it
            # was derived from (not necessarily the whole table).
            try:
                if not is_table(parsed_obj) and not isinstance(
                    parsed_obj, pd.Series
                ):
                    raise SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=(
                            f"Column parser "
                            f"{getattr(parser, 'name', parser_index)!r} "
                            "must return a pandas Series or DataFrame, got "
                            f"{type(parsed_obj).__name__}."
                        ),
                        failure_cases=type(parsed_obj).__name__,
                        check=getattr(
                            parser, "name", f"parser_{parser_index}"
                        ),
                        reason_code=SchemaErrorReason.CHECK_ERROR,
                    )
                if is_table(parsed_obj):
                    if is_table(before):
                        reference = before
                    else:
                        # a table produced from a field must keep the table rows
                        reference = check_obj
                    if validation_context is not None:
                        validation_context.assert_parser_preserves_rows(
                            reference,
                            parsed_obj,
                            getattr(parser, "name", f"parser_{parser_index}"),
                        )
                    elif (
                        len(parsed_obj) != len(reference)
                        or not parsed_obj.index.equals(reference.index)
                    ):
                        raise SchemaError(
                            schema=schema,
                            data=reference,
                            message=(
                                f"Column parser "
                                f"{getattr(parser, 'name', parser_index)!r} "
                                "changed the number of rows or the index "
                                "sequence of column "
                                f"{column_name!r}."
                            ),
                            failure_cases=getattr(
                                parser, "name", f"parser_{parser_index}"
                            ),
                            check=getattr(
                                parser, "name", f"parser_{parser_index}"
                            ),
                            reason_code=SchemaErrorReason.CHECK_ERROR,
                        )
                elif isinstance(parsed_obj, pd.Series):
                    # the parser input is the Series itself on the first hop,
                    # or whatever object the previous parser returned.
                    reference_index = (
                        before.index
                        if isinstance(before, pd.Series)
                        else before[column_name].index
                    )
                    if len(parsed_obj) != len(reference_index) or not (
                        parsed_obj.index.equals(reference_index)
                    ):
                        raise SchemaError(
                            schema=schema,
                            data=check_obj,
                            message=(
                                f"Column parser "
                                f"{getattr(parser, 'name', parser_index)!r} "
                                "returned a Series with a number of rows or an "
                                "index sequence that does not match column "
                                f"{column_name!r}."
                            ),
                            failure_cases=getattr(
                                parser, "name", f"parser_{parser_index}"
                            ),
                            check=getattr(
                                parser, "name", f"parser_{parser_index}"
                            ),
                            reason_code=SchemaErrorReason.CHECK_ERROR,
                        )
            except SchemaError as err:
                return None, self._column_parser_error(
                    schema,
                    check_obj,
                    column_name,
                    parser,
                    parser_index,
                    err,
                    validation_context,
                    error_handler,
                )
        return parsed_obj, None

    def _column_parser_error(
        self,
        schema,
        check_obj,
        column_name,
        parser,
        parser_index,
        original_exc,
        validation_context,
        error_handler,
    ) -> SchemaError:
        """Build, collect and return the SchemaError for a failed column parser."""
        if isinstance(original_exc, SchemaError):
            exc = original_exc
            exc.column_name = column_name
            exc.parser = exc.parser or parser
            exc.parser_index = (
                exc.parser_index if exc.parser_index is not None
                else parser_index
            )
            if exc.reason_code is None:
                exc.reason_code = SchemaErrorReason.CHECK_ERROR
            reason_code = exc.reason_code
            chained = getattr(exc, "original_exc", None)
        else:
            err_msg = (
                f'"{original_exc.args[0]}"'
                if original_exc.args
                else ""
            )
            err_str = f"{original_exc.__class__.__name__}({err_msg})"
            reason_code = SchemaErrorReason.CHECK_ERROR
            exc = SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    f"Error while executing parser "
                    f"{getattr(parser, 'name', parser_index)!r} on column "
                    f"{column_name!r}: {err_str}\n"
                    f"{traceback.format_exc()}"
                ),
                failure_cases=err_str,
                check=getattr(parser, "name", f"parser_{parser_index}"),
                parser=parser,
                parser_index=parser_index,
                reason_code=reason_code,
                column_name=column_name,
                original_exc=original_exc,
            )
            chained = original_exc

        # parser failures cannot be repaired by dropping rows and cannot be
        # attributed to concrete physical rows.
        error_handler.mark_non_row_error(exc)
        error_handler.collect_error(
            validation_type(reason_code),
            reason_code,
            exc,
            chained,
        )
        return exc

    @staticmethod
    def _check_only_component(schema):
        """Shallow-copy a column schema keeping only its checks."""
        check_only = copy_module.copy(schema)
        check_only.parsers = []
        check_only.checks = list(schema.checks)
        return check_only

    def get_regex_columns(self, schema, check_obj) -> Iterable:
        """Get matching column names based on regex column name pattern.

        :param schema: schema specification to use
        :param columns: columns to regex pattern match
        :returns: matching columns
        """
        columns = check_obj.columns
        if isinstance(schema.name, tuple):
            # handle MultiIndex case
            if len(schema.name) != columns.nlevels:
                raise IndexError(
                    f"Column regex name='{schema.name}' is a tuple, expected a "
                    f"MultiIndex columns with {len(schema.name)} number of "
                    f"levels, found {columns.nlevels} level(s)"
                )
            matches = np.ones(len(columns)).astype(bool)
            for i, name in enumerate(schema.name):
                matched = pd.Index(
                    columns.get_level_values(i).astype(str).str.match(name)
                ).fillna(False)
                matches = matches & np.array(matched.tolist())
            column_keys_to_check = columns[matches]
        else:
            if is_multiindex(columns):
                raise IndexError(
                    f"Column regex name {schema.name} is a string, expected a "
                    "dataframe where the index is a pd.Index object, not a "
                    "pd.MultiIndex object"
                )
            column_keys_to_check = columns[
                # str.match will return nan values when the index value is
                # not a string.
                pd.Index(columns.astype(str).str.match(schema.name))
                .fillna(False)
                .tolist()
            ]
        if column_keys_to_check.shape[0] == 0:
            raise SchemaError(
                schema=schema,
                data=columns,
                message=(
                    f"Column regex name='{schema.name}' did not match any "
                    "columns in the dataframe. Update the regex pattern so "
                    f"that it matches at least one column:\n{columns.tolist()}",
                ),
                failure_cases=str(columns.tolist()),
                check=f"no_regex_column_match('{schema.name}')",
                reason_code=SchemaErrorReason.INVALID_COLUMN_NAME,
            )
        # drop duplicates to account for potential duplicated columns in the
        # dataframe.
        return column_keys_to_check.drop_duplicates()

    def coerce_dtype(
        self,
        check_obj: Union[pd.DataFrame, pd.Series],
        schema=None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """Coerce dtype of a column, handling duplicate column names."""
        # pylint: disable=super-with-arguments
        # pylint: disable=fixme
        # TODO: use singledispatchmethod here
        if is_field(check_obj) or is_index(check_obj):
            return super().coerce_dtype(
                check_obj,
                schema=schema,
            )
        return check_obj.apply(
            lambda x: super(ColumnBackend, self).coerce_dtype(
                x,
                schema=schema,
            ),
            axis="columns",
        )

    def run_checks(
        self,
        check_obj,
        schema,
        validation_context=None,
        head=None,
        tail=None,
        sample=None,
        random_state=None,
    ):
        check_results: List[CoreCheckResult] = []
        for check_index, check in enumerate(schema.checks):
            check_args = [None] if is_field(check_obj) else [schema.name]
            try:
                check_results.append(
                    self.run_check(
                        check_obj,
                        schema,
                        check,
                        check_index,
                        *check_args,
                        validation_context=validation_context,
                        head=head,
                        tail=tail,
                        sample=sample,
                        random_state=random_state,
                    )
                )
            except SchemaError as err:
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=err.reason_code
                        or SchemaErrorReason.DATAFRAME_CHECK,
                        message=str(err),
                        failure_cases=err.failure_cases,
                        original_exc=getattr(err, "original_exc", None) or err,
                        row_positions=getattr(err, "row_positions", None),
                    )
                )
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
                err_str = f"{err.__class__.__name__}({ err_msg})"
                msg = (
                    f"Error while executing check function: {err_str}\n"
                    + traceback.format_exc()
                )
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_ERROR,
                        message=msg,
                        failure_cases=err_str,
                        original_exc=err,
                    )
                )
        return check_results


class IndexBackend(ArraySchemaBackend):
    """Backend implementation for pandas index."""

    def validate(
        self,
        check_obj: Union[pd.DataFrame, pd.Series],
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
        validation_context=None,
    ) -> Union[pd.DataFrame, pd.Series]:
        if is_multiindex(check_obj.index):
            raise SchemaError(
                schema,
                check_obj,
                "Attempting to validate mismatch index",
                reason_code=SchemaErrorReason.MISMATCH_INDEX,
            )

        error_handler = ErrorHandler(
            lazy, validation_context=validation_context
        )

        if schema.coerce:
            try:
                check_obj.index = schema.coerce_dtype(check_obj.index)
            except SchemaError as exc:
                error_handler.collect_error(
                    validation_type(exc.reason_code),
                    exc.reason_code,
                    exc,
                )

        try:
            _validated_obj = super().validate(
                check_obj.index.to_series().reset_index(drop=True),
                schema,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )
            assert is_field(_validated_obj)
        except SchemaError as exc:
            error_handler.collect_error(
                validation_type(exc.reason_code),
                exc.reason_code,
                exc,
            )
        except SchemaErrors as exc:
            error_handler.collect_errors(exc.schema_errors, exc)

        if lazy and error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        return check_obj


class MultiIndexBackend(DataFrameSchemaBackend):
    """Backend implementation for pandas multiindex."""

    def coerce_dtype(  # type: ignore[override]
        self,
        # pylint: disable=fixme
        # TODO: make MultiIndex not inherit from DataFrameSchemaBackend
        check_obj: pd.MultiIndex,
        schema=None,
    ) -> pd.MultiIndex:
        """Coerce type of a pd.Series by type specified in dtype.

        :param obj: multi-index to coerce.
        :returns: ``MultiIndex`` with coerced data type
        """
        assert schema is not None, "The `schema` argument must be provided."

        if not schema.coerce:
            return check_obj

        error_handler = ErrorHandler(lazy=True)

        # construct MultiIndex with coerced data types
        coerced_multi_index = {}
        for i, index in enumerate(schema.indexes):
            if all(x is None for x in schema.names):
                index_levels = [i]
            else:
                index_levels = [
                    i
                    for i, name in enumerate(check_obj.names)
                    if name == index.name
                ]
            for index_level in index_levels:
                index_array = check_obj.get_level_values(index_level)
                if index.coerce or schema._coerce:
                    try:
                        _index = deepcopy(index)
                        _index.coerce = True
                        index_array = _index.coerce_dtype(index_array)
                    except SchemaError as err:
                        error_handler.collect_error(
                            validation_type(
                                SchemaErrorReason.DATATYPE_COERCION
                            ),
                            SchemaErrorReason.DATATYPE_COERCION,
                            err,
                        )
                coerced_multi_index[index_level] = index_array

        if error_handler.collected_errors:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=check_obj,
            )

        multiindex_cls = pd.MultiIndex
        # NOTE: this is a hack to support pyspark.pandas
        if type(check_obj).__module__.startswith("pyspark.pandas"):
            # pylint: disable=import-outside-toplevel
            import pyspark.pandas as ps

            multiindex_cls = ps.MultiIndex

        return multiindex_cls.from_arrays(
            [
                # v.dtype may be different than 'object'.
                # - Reuse the original index array to keep the specialized dtype:
                #   v.to_numpy()  converts the array dtype to array of 'object' dtype.
                #   Thus removing the specialized index dtype required to pass a schema's
                #   index specialized dtype : eg: pandera.typing.Index(pandas.Int64Dtype)
                # - For Pyspark only, use to_numpy(), with the effect of keeping the
                #   bug open on this execution environment: At the time of writing, pyspark
                #   v3.3.0 does not provide a working implementation of v.array
                (
                    v.to_numpy()
                    if type(v).__module__.startswith("pyspark.pandas")
                    else v.array
                )
                for _, v in sorted(
                    coerced_multi_index.items(), key=lambda x: x[0]
                )
            ],
            names=check_obj.names,
        )

    def validate(
        self,
        check_obj: Union[pd.DataFrame, pd.Series],
        schema,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
        validation_context=None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """Validate DataFrame or Series MultiIndex.

        :param check_obj: pandas DataFrame of Series to validate.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        :returns: validated DataFrame or Series.
        """
        if schema.coerce:
            check_obj.index = self.__coerce_index(check_obj, schema, lazy)

        # Prevent data type coercion when the validate method is called because
        # it leads to some weird behavior when calling coerce_dtype within the
        # DataFrameSchema.validate call. Need to fix this by having MultiIndex
        # not inherit from DataFrameSchema.
        schema_copy = deepcopy(schema)
        schema_copy.coerce = False
        for index in schema_copy.indexes:
            index.coerce = False

        # rename integer-based column names in case of duplicate index names,
        # with at least one named index.
        if (
            not all(x is None for x in check_obj.index.names)
            and len(set(check_obj.index.names)) != check_obj.index.nlevels
        ):
            index_names = []
            for i, name in enumerate(check_obj.index.names):
                name = i if name is None else name
                if name not in index_names:
                    index_names.append(name)

            columns = {}
            for name, (_, column) in zip(
                index_names, schema_copy.columns.items()
            ):
                columns[name] = column.set_name(name)
            schema_copy.columns = columns
        try:
            validation_result = super().validate(
                self.__to_dataframe(check_obj.index),
                schema_copy,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )
        except SchemaErrors as err:
            # This is a hack to re-raise the SchemaErrors exception and change
            # the schema context to MultiIndex. This should be fixed by with
            # a more principled schema class hierarchy.
            schema_errors = []
            for schema_error in err.schema_errors:
                if is_table(schema_error.failure_cases):
                    failure_cases = schema_error.failure_cases.assign(
                        column=schema_error.schema.name
                    )
                else:
                    failure_cases = schema_error.failure_cases
                schema_errors.append(
                    SchemaError(
                        schema,
                        check_obj,
                        schema_error.args[0],
                        failure_cases,
                        schema_error.check,
                        schema_error.check_index,
                        reason_code=schema_error.reason_code,
                    )
                )

            raise SchemaErrors(
                schema=schema,
                schema_errors=schema_errors,
                data=check_obj,
            ) from err

        assert is_table(validation_result)
        return check_obj

    def __to_dataframe(self, multiindex):
        """
        Emulate the behavior of pandas.MultiIndex.to_frame, but preserve
        duplicate index names if they exist.
        """
        # NOTE: this is a hack to support pyspark.pandas
        if type(multiindex).__module__.startswith("pyspark.pandas"):
            df = multiindex.to_frame()
        else:
            df = pd.DataFrame(
                {
                    i: multiindex.get_level_values(i)
                    for i in range(multiindex.nlevels)
                }
            )
            df.columns = [
                i if name is None else name
                for i, name in enumerate(multiindex.names)
            ]
            df.index = multiindex
        return df

    def __coerce_index(self, check_obj, schema, lazy):
        """Coerce index"""
        try:
            return self.coerce_dtype(
                check_obj.index, schema=schema  # type: ignore [arg-type]
            )
        except SchemaErrors as err:
            if lazy:
                raise
            raise err.schema_errors[0] from err
